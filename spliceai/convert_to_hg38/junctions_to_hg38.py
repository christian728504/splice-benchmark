from pyliftover import LiftOver
import pandas as pd
import sys

def lift_coordinates(input_file, output_file):
    """
    Lift over coordinates from hg19 to hg38.
    Handles TSV input/output using pandas.
    """
    # Initialize liftover
    lo = LiftOver('hg19', 'hg38')
    
    # Read TSV file
    df = pd.read_csv(input_file, sep='\t')
    
    # Create new dataframe for results
    new_records = []
    failed_conversions = []
    
    for idx, row in df.iterrows():
        name = row['#name']
        chrom = row['chrom']
        strand = row['strand']
        txStart = int(row['txStart'])
        txEnd = int(row['txEnd'])
        exonEnds = [int(x) for x in row['exonEnds'].strip(',').split(',') if x]
        exonStarts = [int(x) for x in row['exonStarts'].strip(',').split(',') if x]
        
        # Convert coordinates
        new_txStart = lo.convert_coordinate(chrom, txStart)
        new_txEnd = lo.convert_coordinate(chrom, txEnd)
        
        # Check if main coordinates converted successfully
        if not new_txStart or not new_txEnd:
            failed_conversions.append(f"{name}: Failed to convert transcript boundaries")
            continue
        
        # Convert exon coordinates
        new_exonStarts = []
        new_exonEnds = []
        exon_conversion_failed = False
        
        for start, end in zip(exonStarts, exonEnds):
            new_start = lo.convert_coordinate(chrom, start)
            new_end = lo.convert_coordinate(chrom, end)
            
            if not new_start or not new_end:
                failed_conversions.append(f"{name}: Failed to convert exon coordinates")
                exon_conversion_failed = True
                break
            
            new_exonStarts.append(str(new_start[0][1]))
            new_exonEnds.append(str(new_end[0][1]))
        
        if exon_conversion_failed:
            continue
        
        # Add to new records
        new_records.append({
            '#name': name,
            'chrom': chrom,
            'strand': strand,
            'txStart': new_txStart[0][1],
            'txEnd': new_txEnd[0][1],
            'exonEnds': ','.join(new_exonEnds) + ',',
            'exonStarts': ','.join(new_exonStarts) + ','
        })
    
    # Create new dataframe and write output
    new_df = pd.DataFrame(new_records)
    new_df.to_csv(output_file, sep='\t', index=False)
    
    # Report results
    print(f"\nProcessed {len(df)} entries")
    print(f"Successfully converted {len(new_records)} entries")
    print(f"Failed to convert {len(failed_conversions)} entries")
    
    if failed_conversions:
        print("\nWarning: Some conversions failed:")
        for failure in failed_conversions:
            print(failure)