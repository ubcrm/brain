import imageio
import os

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = 'c:\\projects\\hc2\\' #Add path here
    # r=root, d=directories, f = files

    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.png' in file:
                files.append(os.path.join(r, file))

    with imageio.get_writer(path+"movie.gif", mode='I') as writer:
        for file in files:
            image = imageio.imread(file)
            writer.append_data(image)