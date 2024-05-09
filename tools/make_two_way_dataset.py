with open('../tsp50_train_small.txt', 'r') as file:
    lines = file.readlines()

rearranged_lines = []
for idx, line in enumerate(lines):
    
    if idx % 2 ==0:
        continue
    
    tokens = line.split()
    output_index = tokens.index("output")
    numbers = tokens[output_index + 1:]
    numbers.reverse()
    tokens = tokens[:output_index + 1] + numbers
    rearranged_line = " ".join(tokens)
    rearranged_lines.append(rearranged_line)
    rearranged_lines.append("\n")
    rearranged_lines.append(line)

with open('../tsp50_train_small_2way.txt', 'w') as file:
    file.writelines(rearranged_lines)