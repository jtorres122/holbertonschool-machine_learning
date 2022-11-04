#!/usr/bin/env python3
'''Module contains the inverse(matrix) function'''


def inverse(matrix):
    '''Function calculates the inverse of a matrix'''
    adj = adjugate(matrix)
    det = determinant(matrix)
    if det == 0:
        return None
    return [[y / det for y in x] for x in adj]


def adjugate(matrix):
    '''Function calculates the adjugate matrix of a matrix'''
    cof = minor(matrix)
    temp = []
    for x in range(len(cof)):
        temp.append([])
        for y in range(len(cof)):
            temp[x].append(cof[y][x])
    return temp


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
            sign = (-1) ** ((x + y) % 2)
            t.append(determinant(s) * sign)
        mino.append(t)
    return mino


def determinant(matrix):
    '''Function calculates the determinant of a matrix'''
    det = 0
    if matrix == [[]]:
        return 1
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        det = (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
        return det
    for x, num in enumerate(matrix):
        temp = []
        P = matrix[0][x]
        for row in matrix[1:]:
            new = []
            for j in range(len(matrix)):
                if j != x:
                    new.append(row[j])
            temp.append(new)
        det += P * determinant(temp) * (-1) ** x
    return det
