#!/usr/bin/env python3
'''Module contains the minor(matrix) function'''


def minor(matrix):
    '''Function calculates the minor matrix of a matrix'''
    if len(matrix) == 1:
        return [[1]]
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')
    if any(not isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError('matrix must be a non-empty square matrix')
    mino = []
    for x in range(len(matrix)):
        t = []
        for y in range(len(matrix[0])):
            s = []
            for row in (matrix[:x] + matrix[x + 1:]):
                s.append(row[:y] + row[y + 1:])
            t.append(determinant(s))
        mino.append(t)
    return mino


def determinant(matrix):
    '''Function calculates the determinant of a matrix'''
    det = 0
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        det = (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
        return det
    for x, num in enumerate(matrix[0]):
        row = [r for r in matrix[1:]]
        temp = []
        for r in row:
            a = []
            for c in range(len(matrix)):
                if c != x:
                    a.append(r[c])
            temp.append(a)
        det += num * (-1) ** x * determinant(temp)
    return det
