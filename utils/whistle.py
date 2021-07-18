import numpy as np
import sys

sys.path.append("../../utils")
import utils as utils
import pdb


class Envelope:
    def __init__(self, target=0.0, value=0.0, rate=0.001, state=0):
        self.target = target
        self.value = value
        self.rate = rate
        self.state = state

    def set_target(self, target):
        self.target = target
        if self.value != self.target:
            self.state = 1

    def tick(self):
        if self.state:
            if self.target > self.value:
                self.value += self.rate
                if self.value >= self.target:
                    self.value = self.target
                    self.state = 0
            else:
                self.value -= self.rate
                if self.value <= self.target:
                    self.value = self.target
                    self.state = 0


class Noise:
    def __init__(self, seed=0.0, value=0.0, rand_max=30):
        self.seed = seed
        self.value = value
        self.rand_max = rand_max

    def tick(self):
        self.value = 3.0 * np.random.random() / (self.rand_max + 1) - 1.0


class OnePole:
    def __init__(self, a=0, b=0, gain=1.0, inputs=0.0, outputs=0.0):
        self.a = a
        self.b = b
        self.gain = gain
        self.inputs = inputs
        self.outputs = outputs

    def set_pole(self, pole):
        if pole > 0.0:
            self.a = 1.0 - pole
        else:
            self.a = 1.0 + pole
        self.b = -pole

    def tick(self, input_):
        self.inputs = self.gain * input_
        self.outputs = self.a * self.inputs - self.b * self.outputs


class SineWave:
    def __init__(
        self,
        sample_rate=44100,
        table_size=22050,
        time=0.0,
        rate=1.0,
        phase_offset=0.0,
        i_index=0,
        alpha=0.0,
    ):
        self.sample_rate = sample_rate
        self.table_size = table_size
        self.time = time
        self.phase_offset = phase_offset
        self.i_index = i_index
        self.alpha = alpha

        self.table = np.zeros(self.table_size)
        temp = 1.0 / self.table_size
        for i in range(self.table_size):
            self.table[i] = np.sin(2 * np.pi * i * temp)

    def reset(self):
        self.time = 0

    def set_frequency(self, freq):
        self.frequency = freq

    def set_rate(self, freq):
        self.rate = self.table_size * freq / self.sample_rate

    def add_time(self, time):
        self.time += time

    def add_phase(self, phase_offset):
        self.time += (phase_offset - self.phase_offset) * self.table_size
        self.phase_offset = phase_offset

    def tick(self):
        while self.time < 0.0:
            self.time += self.table_size
        while self.time >= self.table_size:
            self.time -= self.table_size
        self.i_index = np.floor(self.time)  # +1
        self.alpha = self.time - self.i_index

        self.tmp = self.table[int(self.i_index)]
        self.tmp += self.alpha * (self.table[int(self.i_index)] - self.tmp)
        self.time += self.rate


class Sphere:
    def __init__(self, radius=1):
        self.radius = radius
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)

    def set_radius(self, radius):
        self.radius = radius

    def get_relative_position(self, position):
        return position - self.position

    def is_inside(self, position):
        return np.linalg.norm(self.get_relative_position(position), 2) - self.radius

    def add_velocity(self, velocity):
        self.velocity += velocity

    def tick(self, dt):
        self.position += dt * self.velocity


class Whistle:
    def __init__(
        self,
        can_radius=200,
        pea_radius=50,
        bump_radius=150,
        norm_can_loss=0.97,
        gravity=20.0,
        norm_tick_size=0.004,
        env_rate=0.001,
        sample_rate=44100,
        fipple_freq_mod=0.125,
        fipple_gain_mod=0.125,
        blow_freq_mod=0.5,
        noise_gain=0.5,
        base_freq=2500,
        sine_rate=2500,
        pole=0.95,
    ):
        self.can_radius = can_radius
        self.pea_radius = pea_radius
        self.bump_radius = bump_radius
        self.norm_can_loss = norm_can_loss
        self.gravity = gravity
        self.norm_tick_size = norm_tick_size
        self.env_rate = env_rate
        self.sample_rate = sample_rate

        self.temp_vec_p = np.zeros(3)
        self.temp_vec = np.zeros(3)

        self.one_pole = OnePole()
        self.noise = Noise()
        self.env = Envelope()
        self.can = Sphere()
        self.pea = Sphere()
        self.bumper = Sphere()
        self.sine = SineWave()

        # sine wave settings
        self.sine.set_rate(sine_rate)

        # init can
        self.can.set_radius(self.can_radius)
        self.can.position = np.zeros(3)
        self.can.velocity = np.zeros(3)

        # init pole
        self.one_pole.set_pole(pole)

        # init bumper
        self.bumper.set_radius(self.bump_radius)
        self.bumper.position = np.array([0.0, self.can_radius - self.bump_radius, 0.0])
        self.bumper.velocity = np.zeros(3)

        # init pea
        self.pea.set_radius(self.pea_radius)
        self.pea.position = np.array([0.0, self.can_radius / 2.0, 0.0])
        self.pea.velocity = np.array([35.0, 15.0, 0.0])

        # init envelope
        self.env.rate = self.env_rate
        self.env.set_target(1)

        # init blow settings
        self.fipple_freq_mod = fipple_freq_mod
        self.fipple_gain_mod = fipple_gain_mod
        self.blow_freq_mod = noise_gain
        self.noise_gain = noise_gain
        self.base_freq = base_freq

        self.tick_size = self.norm_tick_size
        self.can_loss = self.norm_can_loss

        self.sub_sample = 1
        self.sub_sample_count = self.sub_sample
        self.frame_count = 0

        self.last_frame = 0

    def set_frequency(self, freq):
        self.base_freq = freq * 4  # whistle is a transposing insturment

    def start_blowing(self, amplitude, rate):
        # check of amp or rate is >= 0 -> throw error
        self.env.rate = rate
        self.env.set_target(amplitude)

    def stop_blowing(self, rate):
        self.env.rate = rate
        self.env.set_target(0.0)

    def note_on(self, freq, amplitude):
        self.set_frequency(freq)
        self.start_blowing(amplitude * 2.0, amplitude * 0.2)

    def note_off(self, amplitude):
        self.stop_blowing(amplitude * 0.02)

    def tick(self):
        gain = 0.5
        mod = 0.0
        env_out, tempX, tempY = 0, 0, 0

        self.sub_sample_count -= 1
        if self.sub_sample_count <= 0:
            self.temp_vec_p = self.pea.position
            self.sub_sample_count = self.sub_sample
            temp = self.bumper.is_inside(self.temp_vec_p)
            self.frame_count += 1

            if self.frame_count >= (1470 / self.sub_sample):
                self.frame_count = 0

            self.env.tick()
            env_out = self.env.value

            if temp < (self.bump_radius + self.pea_radius):
                self.noise.tick()
                tempX = env_out * self.tick_size * 2000 * self.noise.value
                self.noise.tick()
                tempY = -env_out * self.tick_size * 1000 * (1.0 - self.noise.value)
                self.pea.add_velocity(np.array([tempX, tempY, 0]))
                self.pea.tick(self.tick_size)

            mod = np.exp(-temp * 0.01)  # exp distance fall off of fipple/pea effect
            self.one_pole.tick(mod)
            temp = self.one_pole.outputs
            gain = (1.0 - (self.fipple_gain_mod * 0.5)) + (
                2.0 * self.fipple_gain_mod * temp
            )
            gain = np.power(gain, 2)  # squared distance gain
            temp_freq = (
                1.0
                + self.fipple_freq_mod * (0.25 - temp)
                + self.blow_freq_mod * (env_out - 1.0)
            )
            temp_freq *= self.base_freq
            self.sine.set_frequency(temp_freq)

            self.temp_vec_p = self.pea.position
            temp = self.can.is_inside(self.temp_vec_p)
            temp = -temp

            if temp < (self.pea_radius * 1.25):
                self.temp_vec = self.pea.velocity
                tempX = self.temp_vec_p[0]
                tempY = self.temp_vec_p[1]
                phi = -np.arctan2(tempY, tempX)

                cosphi = np.cos(phi)
                sinphi = np.sin(phi)
                temp1 = (cosphi * self.temp_vec[0]) - (sinphi * self.temp_vec[1])
                temp2 = (sinphi * self.temp_vec[0]) + (cosphi * self.temp_vec[1])
                temp1 = -temp1
                tempX = (cosphi * temp1) + (sinphi * temp2)
                tempY = (-sinphi * temp1) + (cosphi * temp2)
                self.pea.velocity = np.array([tempX, tempY, 0])
                self.pea.tick(self.tick_size)
                self.pea.velocity = np.array(
                    [tempX * self.can_loss, tempY * self.can_loss, 0.0]
                )
                self.pea.tick(self.tick_size)

                temp = np.linalg.norm(self.temp_vec_p, 2)
                if temp > 0.01:
                    tempX = self.temp_vec_p[0]
                    tempY = self.temp_vec_p[1]

                    phi = np.arctan2(tempY, tempX)
                    phi += 0.3 * temp / self.can_radius
                    cosphi = np.cos(phi)
                    sinphi = np.sin(phi)
                    tempX = 3.0 * temp * cosphi
                    tempY = 3.0 * temp * sinphi
                else:
                    tempX = 0.0
                    tempY = 0.0

                self.noise.tick()
                temp = (
                    (0.9 + 0.1 * self.sub_sample * self.noise.value)
                    * env_out
                    * 0.6
                    * self.tick_size
                )
                self.pea.add_velocity(
                    np.array(
                        [
                            temp * tempX,
                            (temp * tempY) - (self.gravity * self.tick_size),
                            0,
                        ]
                    )
                )
                self.pea.tick(self.tick_size)
            temp = np.power(env_out, 2) * gain / 2
            self.sine.tick()
            self.noise.tick()
            soundMix = temp * (self.sine.tmp + (self.noise_gain * self.noise.value))
            self.last_frame = 0.20 * soundMix
