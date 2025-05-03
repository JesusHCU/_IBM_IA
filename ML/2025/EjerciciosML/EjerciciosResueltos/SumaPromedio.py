def calcular_suma_y_promedio(lista_numeros):
    suma = 0
    cont = 0
    numero_ele = len(lista_numeros)
    if(numero_ele==0):
        resultado={"suma":suma, "promedio":suma/1}
    else:
        for numero in lista_numeros:
            suma = suma + numero
            cont = cont + 1
        resultado={"suma":suma, "promedio":suma/numero}
    return (resultado)