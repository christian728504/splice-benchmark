import numpy as np

def append_cpm_to_bed(bed_file, output_file=None):
    """
    Calculate cpm for junctions and append it to the original BED file
    
    Args:
        bed_file: Path to input junctions BED file
        output_file: Path for output file. If None, will append '_with_cpm' to input filename
    """
    if output_file is None:
        output_file = bed_file.replace('.bed', '_with_cpm.bed')
        if output_file == bed_file:  # if no .bed extension
            output_file = bed_file + '_with_cpm'
    
    # First pass: collect counts and lengths for cpm calculation
    junctions = []
    bed_lines = []
    
    with open(bed_file, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            bed_lines.append(fields)
            
            # Get count from 5th column
            count = int(fields[4])
            
            # Calculate junction length
            block_sizes = [int(x) for x in fields[-2].split(',') if x]
            block_starts = [int(x) for x in fields[-1].split(',') if x]
            intron_length = block_starts[1] - block_sizes[0]
            total_length = sum(block_sizes) + intron_length
            
            junctions.append((count, total_length))
    
    # Convert to numpy arrays for efficient calculation
    counts, lengths = np.array(junctions).T
    
    # Calculate cpm
    rate = np.log1p(counts) - np.log(lengths)
    denom = np.log(np.sum(np.exp(rate)))
    cpm = np.exp(rate - denom + np.log(1e6))
    
    # Write output with cpm appended
    with open(output_file, 'w') as f:
        for i, fields in enumerate(bed_lines):
            # Append cpm to the original BED line
            new_line = fields + [f"{cpm[i]:.6f}"]
            f.write('\t'.join(str(x) for x in new_line) + '\n')
    
    return output_file