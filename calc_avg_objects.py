from collections import defaultdict

def read_files(play_file, count_file):
    play_types = []
    total_counts = []  # Store total counts
    
    with open(play_file, 'r') as f:
        for line in f:
            play_types.append(line.strip())
    
    # Read total count from the second file
    with open(count_file, 'r') as f:
        for line in f:
            _, count = line.strip().split(',')  # Ignore dummy variable
            total_counts.append(float(count))
    
    # Check if both files have the same number of lines
    if len(play_types) != len(total_counts):
        raise ValueError("Files must have the same number of lines")
    
    # Create defaultdict to store total counts and occurrence count
    play_totals = defaultdict(lambda: [0, 0])  # [sum of counts, occurrence count]
    
    for play, count in zip(play_types, total_counts):
        play_totals[play][0] += count  # Sum of counts
        play_totals[play][1] += 1  # Count occurrences
    
    # Compute the average total count for each play type
    play_averages = {play: total / count for play, (total, count) in play_totals.items()}
    
    return play_averages

# Example usage
play_file = "label_results.txt"  # File with play types
count_file = "detection_results.txt"  # File with total counts and dummy variable
averages = read_files(play_file, count_file)

# Print results
for play, avg in averages.items():
    print(f"{play}: {avg:.2f}")