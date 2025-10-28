import numpy as np
from sacn import sACNsender
import math
class SACNPixelSender:
    def __init__(self, receivers,start_universe=1):
        """
        Initialize the SACNPixelSender with receiver configurations.
        :param receivers: List of dicts, each with 'ip', 'pixel_count', and 'addressing_array' keys.
        """
        self.receivers = receivers
        self.sender = sACNsender()
        
        # Enable manual flush for synchronized sending
        self.sender.manual_flush = True
        self.sender.start()

        # Set up universes for each receiver
        self.receiver_universes = []
        universe_counter = start_universe
        for receiver in receivers:
            universe_count = math.ceil(receiver['pixel_count'] / 170)
            receiver_universes = list(range(universe_counter, universe_counter + universe_count))
            self.receiver_universes.append(receiver_universes)
            universe_counter += universe_count

            # Activate universes for this receiver
            for universe in receiver_universes:
                self.sender.activate_output(universe)
                self.sender[universe].destination = receiver['ip']

    def create_mask(self, height, width):
        """
        Creates a binary mask showing which pixels are mapped by receivers.
        
        :param height: Height of the source image
        :param width: Width of the source image
        :return: numpy array of shape (height, width) with 1s where pixels are mapped
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for receiver in self.receivers:
            # Clip coordinates to valid image boundaries
            x_coords = np.clip(receiver['addressing_array'][:, 0], 0, height - 1)
            y_coords = np.clip(receiver['addressing_array'][:, 1], 0, width - 1)
            
            # Set mapped pixels to 1
            mask[x_coords, y_coords] = 1
            
        return mask

    def send(self, source_array):
        """
        Send pixel data to all configured receivers based on their addressing arrays.
        :param source_array: numpy array of shape (width, height, 3) containing source pixel data.
        """
        height, width, _ = source_array.shape
        
        # Prepare all universe data first (before sending anything)
        for receiver, universes in zip(self.receivers, self.receiver_universes):
            # Vectorized extraction of pixel data
            x_coords = np.clip(receiver['addressing_array'][:, 0], 0, height - 1)
            y_coords = np.clip(receiver['addressing_array'][:, 1], 0, width - 1)
            receiver_data = source_array[x_coords, y_coords]

            # Prepare data in 170-pixel chunks
            for i, universe in enumerate(universes):
                start = i * 170
                end = min(start + 170, receiver['pixel_count'])
                universe_data = receiver_data[start:end].flatten()
                # Pad the last universe if necessary
                if universe_data.size < 510:
                    universe_data = np.pad(universe_data, (0, 510 - universe_data.size), 'constant')
                
                # Assign data without auto-sending
                self.sender[universe].dmx_data = universe_data.tobytes()
        
        # Now flush all universes at once
        self.sender.flush()

    def close(self):
        """
        Properly close the sACN sender
        """
        self.sender.stop()

    def analyze_row_groups(self, max_pixels_per_group=170):
        """
        Analyze and group pixels in rows that belong to the same receiver.
        
        :param max_pixels_per_group: Maximum number of pixels per group (default 170 for sACN universe limit)
        :return: Dictionary mapping receiver indices to lists of pixel groups
        """
        receiver_groups = {}
        
        for idx, receiver in enumerate(self.receivers):
            # Get coordinates from addressing array
            coordinates = receiver['addressing_array']
            #find the unique rows
            rows = np.unique(coordinates[:, 0])
            #find the number of pixels in each row
            row_counts = {row: np.sum(coordinates[:, 0] == row) for row in rows}         
            #initialize the group list
            groups = []
            #initialize the current group
            current_group = []
            #initialize the current row
            current_row = None
            #iterate through the rows
            group_pixel_count = 0
            pixels_in_group = []
            for row in rows:
                if current_row is None:
                    current_row = row
                #if the row is different from the current row
                row_count = row_counts[row]
                if (group_pixel_count + row_count) <= max_pixels_per_group:
                    #store the current group
                    current_group.append(row)
                    #reset the group count
                    group_pixel_count += row_count
                    #reset the current group
                else:
                    groups.append(current_group)
                    current_group = [row]
                    pixels_in_group.append(group_pixel_count)
                    group_pixel_count = row_count
            #handle the last group
            groups.append(current_group)
            pixels_in_group.append(group_pixel_count)
            print(groups, pixels_in_group,receiver['ip'])

# The rest of the code (generate_frame_data and main function) remains the same

def generate_frame_data():
    """
    Generate random RGB pixel data for each frame.
    In a real application, this would pull data from your actual source.
    """
    width, height = 100, 100  # Example dimensions
    return np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)

def make_indicesHS(filename):
    in_list=np.loadtxt(filename, delimiter=',').tolist()
    indices = []
    for sublist in in_list:       
        if sublist[2]>0:
            for m in range(int(sublist[2])):
                indices.append([sublist[0], m+sublist[1]])
        else:
            for m in range(int(-sublist[2])):
                indices.append([sublist[0], sublist[1]-sublist[2]-1-m])   
    return np.array(indices).astype(int)

def main():
    receivers = [
        {
            'ip': '192.168.68.121',
            'pixel_count': 3500,
            #'addressing_array':make_indicesH()
            'addressing_array':make_indicesHS(r"data.txt")
        }
    ]

    sender = SACNPixelSender(receivers)
    sender.analyze_row_groups(255)
    sender.close()

if __name__ == "__main__":
    main()