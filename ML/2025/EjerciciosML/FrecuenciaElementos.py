def contar_frecuencia(lista):
    frecuencia = {}
    i = 0
    while i < len(lista):
        if lista[i] in frecuencia:
            frecuencia[lista[i]] += 1
        else:
            frecuencia[lista[i]] = 1
        i += 1
    return frecuencia


elementos = [1, 2, 2, 3, 1, 2, 4, 5, 4]
resultado = contar_frecuencia(elementos)
print(resultado)