import pstats
import sys

def analyze_profile(profile_file, filter_path, output_file):
    with open(output_file, 'w') as stream:
        p = pstats.Stats(profile_file, stream=stream)    # the class takes stream as an arg, see: https://docs.python.org/3/library/profile.html
        p.sort_stats('tottime')     # you can choose other sorting options
        p.print_stats(filter_path)      # interactive mode equivalent: stats filter_path

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python analyze_profile.py <profile_file> <filter_path> <output_file>")
        sys.exit(1)

    profile_file = sys.argv[1]
    filter_path = sys.argv[2]
    output_file = sys.argv[3]

    analyze_profile(profile_file, filter_path, output_file)
    print(f"Profile analysis written to {output_file}")