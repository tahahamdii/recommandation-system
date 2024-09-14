def count_uniforms(n, uniforms):
    count = 0
    for i in range(n):
        hc = uniforms[i][0]
        for j in range(n):
            if i != j:
                gc = uniforms[j][1]
                if hc == gc:
                    count += 1
    return count


def main():
    import sysg
    input = sys.stdin.read
    data = input().strip().split('\n')
    print('inputs')
    print(data)

    n = int(data[0])
    print(" numver of teams : {n}")
    uniforms = [tuple(map(int, line.split())) for line in data[1:n+1]]

    print(count_uniforms(n, uniforms))


if __name__ == '__main__':
    main()
