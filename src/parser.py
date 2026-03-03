"""Parses OR-Library SPP file format"""
import numpy as np

def parse_file(filepath):
    with open(filepath, 'r') as f:
        lines= [line.strip() for line in f if line.strip()]
    
    first=lines[0].split()
    num_rows=int(first[0])
    num_cols=int(first[1])

    costs = np.zeros(num_cols, dtype=int)
    matrix=np.zeros((num_rows, num_cols), dtype=int)
        
    for col_num, line in enumerate(lines[1:]):
        nums= list(map(int, line.split()))
        costs[col_num] = nums[0]
        # nums[1] is the number of rows covered by this column and the rest are the rows covered
        for row in nums[2:]:
            matrix[row-1, col_num] = 1
    
    return {"num_rows": num_rows,
            "num_cols": num_cols,
            "costs": costs,
            "matrix": matrix}

# test the parser          
if __name__ == "__main__":
    data = parse_file("data/sppnw41.txt")
    
    print("num_rows:", data["num_rows"])
    print("num_cols:", data["num_cols"])
    print("First 5 costs:", data["costs"][:5])
    print("Coverage per row (first 5):", data["matrix"].sum(axis=1)[:5])
    print("Min coverage per col:", data["matrix"].sum(axis=0).min())