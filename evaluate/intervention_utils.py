import os
import sys
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cwd))


def calculate_quadrant_indices(rows, cols, quadrant):
    """
    Calculate the start and end indices for each quadrant in the flattened tensor.
    """
    row_start, row_end = 0, 7
    col_start, col_end = 0, 7
    
    if quadrant == 2:  # Top Right
        col_start, col_end = 7, 14
    elif quadrant == 3:  # Bottom Left
        row_start, row_end = 7, 14
    elif quadrant == 4:  # Bottom Right
        row_start, row_end = 7, 14
        col_start, col_end = 7, 14

    indices = []
    for row in range(row_start, row_end):
        for col in range(col_start, col_end):
            index = row * rows + col
            indices.append(index)
    
    return indices

q1 = calculate_quadrant_indices(14, 14, 1)
q2 = calculate_quadrant_indices(14, 14, 2)
q3 = calculate_quadrant_indices(14, 14, 3)
q4 = calculate_quadrant_indices(14, 14, 4)

def construct_indices(sampled_patches, granularity, zero_shot):
    
    if granularity==0:
        sampled_patches = sampled_patches.reshape(32,16)
    
        indices = sampled_patches.nonzero()
        expanded_indices = []

        for element in indices:
            layer = int(element[0].item())
            head = int(element[1].item())
            if element[0] < 24:
                for a in range(0, 50 if zero_shot else 148):
                    expanded_indices.append([layer, head, a])
            else:
                for a in range(0, 99 if zero_shot else 197):
                    expanded_indices.append([layer, head, a])
        indices = expanded_indices

        return indices

    if granularity==1:
    
        indices = sampled_patches.nonzero()
        expanded_indices = []

        for element in indices:
            index = int(element.item())
            total_elements_per_layer_first_24_heads = 16 * 2  # 16 layers, 2 elements per layer
            total_elements_per_layer_next_8_heads = 16 * 3  # 16 layers, 3 elements per layer
            total_elements_first_24_heads = 24 * total_elements_per_layer_first_24_heads
            
            if index < total_elements_first_24_heads:
                head = index // total_elements_per_layer_first_24_heads
                layer = (index % total_elements_per_layer_first_24_heads) // 2
                quadrant = index % 2
            else:
                adjusted_index = index - total_elements_first_24_heads
                head = 24 + (adjusted_index // total_elements_per_layer_next_8_heads)
                layer = (adjusted_index % total_elements_per_layer_next_8_heads) // 3
                quadrant = adjusted_index % 3
            
            if quadrant == 0:
                expanded_indices.append([head, layer, 0])
            elif quadrant == 1:
                if head<24:
                    for a in range(1, 50):
                        expanded_indices.append([head,layer, a])
                else:
                    for a in q1:
                        expanded_indices.append([head, layer, a+1])
            elif quadrant == 2:
                for a in q2:
                    expanded_indices.append([head, layer, a+1])
        indices = expanded_indices

        return indices
    
    if granularity==2:
    
        indices = sampled_patches.nonzero()
        expanded_indices = []

        for element in indices:
            index = int(element.item())
            total_elements_per_layer_first_24_heads = 16 * 50  # 16 layers, 2 elements per layer
            total_elements_per_layer_next_8_heads = 16 * 99  # 16 layers, 3 elements per layer
            total_elements_first_24_heads = 24 * total_elements_per_layer_first_24_heads
            
            if index < total_elements_first_24_heads:
                head = index // total_elements_per_layer_first_24_heads
                layer = (index % total_elements_per_layer_first_24_heads) // 50
                quadrant = index % 50
            else:
                adjusted_index = index - total_elements_first_24_heads
                head = 24 + (adjusted_index // total_elements_per_layer_next_8_heads)
                layer = (adjusted_index % total_elements_per_layer_next_8_heads) // 99
                quadrant = adjusted_index % 99
            
            expanded_indices.append([head, layer, quadrant])
        indices = expanded_indices

        return indices