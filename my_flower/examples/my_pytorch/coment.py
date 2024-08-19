print("Please enter your comment. Enter 'END' on a new line to finish.")
lines = []
while True:
    line = input()
    if line == "END":
        break
    lines.append(line)
comment = "\n".join(lines)
# 入力を_README.mdファイルに書き込む
with open("_README.md", "w") as file:
    file.write(comment)
