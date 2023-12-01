import argparse
import matplotlib.pyplot as plt

def plot_data_from_file(filename, num_columns):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Extract headers
    headers = lines[1].strip().split()
    x_label = headers[0]
    y_labels = headers[1:num_columns]

    # Initialize data storage
    x_data = []
    y_data = {y_label: [] for y_label in y_labels}

    # Process data lines
    for line in lines[2:]:
        values = line.strip().split()
        x_data.append(float(values[0]))
        for i, y_label in enumerate(y_labels):
            y_data[y_label].append(float(values[i + 1]))

    # Plotting
    plt.figure(figsize=(12, 6))
    for y_label in y_labels:
        plt.plot(x_data, y_data[y_label], label=y_label)

    plt.xlabel('Matrix Cardinality')
    plt.ylabel('Time in ms')
    plt.title('Matrix Multiplication Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot graph from data file.')
    parser.add_argument('filename', type=str, help='The data file to plot.')
    parser.add_argument('--fast', action='store_true', help='Use 5 columns of data instead of 8.')
    parser.add_argument('--gpuonly', action='store_true', help='Use 3 columns of data instead of 8.')
    args = parser.parse_args()

    # Determine the number of columns based on arguments
    num_columns = 9
    if args.fast:
        num_columns = 6
    elif args.gpuonly:
        num_columns = 4

    plot_data_from_file(args.filename, num_columns)

if __name__ == "__main__":
    main()
