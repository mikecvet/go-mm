import argparse
import matplotlib.pyplot as plt

def plot_data_from_file(filename, mode):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Extract headers
    headers = lines[1].strip().split()
    x_label = headers[0]

    # Initialize data storage
    x_data = []
    y_data = []

    # Process data lines
    for line in lines[2:]:
        values = line.strip().split()
        x_data.append(float(values[0]))

        if mode == 'cmp':
            y_data.append([float(values[4]) / float(values[1]) * 100, 
                           float(values[5]) / float(values[2]) * 100])

    # Plotting
    plt.figure(figsize=(12, 6))
    if mode == 'cmp':
        plt.plot(x_data, [y[0] for y in y_data], label=headers[4] + '/' + headers[1])
        plt.plot(x_data, [y[1] for y in y_data], label=headers[5] + '/' + headers[2])
        plt.ylabel('Relative Completion Time Percentage (%)')
    else:
        # Original plotting logic (as per your script)
        pass

    plt.xlabel(x_label)
    if mode == 'cmp':
        plt.title('Performance Ratio - Hand-Written Go to Metal')
    else:    
        plt.title('Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot graph from data file.')
    parser.add_argument('filename', type=str, help='The data file to plot.')
    parser.add_argument('--cmp', action='store_true', help='Compare hand-written CPU to GPU methods')
    args = parser.parse_args()

    mode = 'cmp' if args.cmp else 'original'

    plot_data_from_file(args.filename, mode)

if __name__ == "__main__":
    main()
