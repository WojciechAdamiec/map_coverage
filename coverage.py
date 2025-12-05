from PIL import Image
from dataclasses import dataclass
from board_generator import prepare_board
from mark_terrain import mark_terrain
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from rich.logging import RichHandler
import os
import logging
import shutil


FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("coverage_logger")
INPUT_DIR = "input"
THRESHOLD = 1
OPTIMAL_COVERAGE = float(33)


@dataclass
class MapImages:
    photo_path: str
    board_path: str
    coverage_path: str


class DataExecutor:
    def __init__(self):
        self.load_data()
    
    def load_data(self):
        self.input_images = dict()
        for filename in os.listdir(INPUT_DIR):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                logger.info(f"Found input image file: {filename}")
                self.input_images[filename] = float(0)

    def execute(self):
        for filename in self.input_images.keys():
            output_dir = os.path.join("output", os.path.splitext(filename)[0])
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)

            src_path = os.path.join(INPUT_DIR, filename)
            self.make_photo_image(src_path, output_dir, filename)
            self.make_board_image(src_path, output_dir, filename)
            self.make_coverage_image(src_path, output_dir, filename)

    def make_photo_image(self, src_path, output_dir, filename):
        shutil.copy(src_path, os.path.join(output_dir, f"photo_{filename}"))

    def make_board_image(self, src_path, output_dir, filename):
        prepare_board(src_path, os.path.join(output_dir, f"board_{filename}"))

    def make_coverage_image(self, src_path, output_dir, filename):
        coverage = mark_terrain(os.path.join(output_dir, f"board_{filename}"), os.path.join(output_dir, f"coverage_{filename}"))
        self.input_images[filename] = coverage

    def display(self):
        for filename in self.input_images.keys():
            output_dir = os.path.join("output", os.path.splitext(filename)[0])
            presenter = DataPresenter()
            presenter.load_images(output_dir)
            logger.info(f"Displayed coverage for {filename}: {self.input_images[filename]:.2f}%")
            presenter.display(coverage=self.input_images[filename])


class DataPresenter:
    def load_images(self, image_dir):
        filenames = []
        for filename in os.listdir(image_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                filenames.append(os.path.join(image_dir, filename))

        self.map_images = MapImages(
            photo_path=self._load_image_type(filenames, "photo"),
            board_path=self._load_image_type(filenames, "board"),
            coverage_path=self._load_image_type(filenames, "coverage"),
        )


    def _load_image_type(self, filenames, type_keyword):
        for filepath in filenames:
            if type_keyword in filepath:
                logger.info(f"Loaded {type_keyword} image: {filepath}")
                return filepath
        logger.error(f"No image found for type: {type_keyword}")
        return None


    def display(self, coverage):
        fig, axes = plt.subplots(1, 3, figsize=(8, 4), dpi=160)
        axes = axes.flatten()

        images = [
            self.map_images.photo_path,
            self.map_images.board_path,
            self.map_images.coverage_path,
        ]

        if coverage >= OPTIMAL_COVERAGE:
            text = f"Terrain Coverage: {coverage:.2f}% Recommended minimum is 33%\nYou have optimal coverage!"
        else:
            percentage_of_optimal = coverage / OPTIMAL_COVERAGE
            relative_missing = 100 / percentage_of_optimal - 100
            text = f"Terrain Coverage: {coverage:.2f}% Recommended minimum is 33%\nAdd additional {relative_missing:.2f}% terrains of what you already have."


        for ax, img_path in zip(axes, images):
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis("off")

        fig.suptitle(text, fontsize=16)

        plt.tight_layout()
        plt.show()


data_executor = DataExecutor()
data_executor.execute()
data_executor.display()