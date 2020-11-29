import imageio
import numpy as np
import os
from pathlib import Path
import re

def count_simulations(file_array):
    pattern = re.compile('Simulation(\d+)')
    nums = [re.match(pattern,f).group() for f in file_array if re.match(pattern,f) != None]
    return list(set(nums))

def find_images_in_path(image_path):
    '''Given a path, finds the simulation output images in that directory and returns
        a list of the relevant file names, separated into batches by simulation number'''
    files = []
    simulation_titles = []
    for r, d, f in os.walk(image_path):
        simulation_titles = count_simulations(f)
        for title in simulation_titles:
            pattern = re.compile(title)
            files.append([os.path.join(r,file) for file in f if re.match(pattern,file)])
    return files, simulation_titles

def make_gif_from_files(files, simulation_titles):
    for i in range(len(files)):
        sim = files[i]
        title = simulation_titles[i]
        with imageio.get_writer(video_path / (title + ".gif"), mode='I') as writer:
            for file in sim:
                image = imageio.imread(file)
                writer.append_data(image)

if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_PATH = Path(ROOT_DIR)
    video_path = ROOT_PATH / 'video'

    image_path = ROOT_PATH / '../sim/renders/camera1/'
    files, simulation_titles = find_images_in_path(image_path)
    simulation_titles = [s+'-camera1' for s in simulation_titles]
    make_gif_from_files(files,simulation_titles)

    image_path = ROOT_PATH / '../sim/renders/camera2/'
    files, simulation_titles = find_images_in_path(image_path)
    simulation_titles = [s+'-camera2' for s in simulation_titles]
    make_gif_from_files(files,simulation_titles)





