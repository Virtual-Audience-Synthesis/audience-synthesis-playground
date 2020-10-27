import scrapy
import requests
from pathlib import Path


class LaughSpider(scrapy.Spider):
    name = "laugh"
    base_url = "https://freesound.org"

    def start_requests(self):
        urls = [
            f"{self.base_url}/search/?q={self.name}&page={index}#sound"
            for index in range(1, 112)
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        laugh_dir = Path(__file__).parent.joinpath(self.name)
        laugh_dir.mkdir(parents=True, exist_ok=True)
        for audio_ref in response.css("a.mp3_file"):
            audio_url = self.base_url + audio_ref.css("a::attr(href)").extract()[0]
            audio_name = audio_ref.css("a::text").extract()[0].replace(" ", "_")
            mp3 = requests.get(audio_url)
            with open(laugh_dir.joinpath(audio_name + ".mp3"), "wb") as mp3_file:
                mp3_file.write(mp3.content)
