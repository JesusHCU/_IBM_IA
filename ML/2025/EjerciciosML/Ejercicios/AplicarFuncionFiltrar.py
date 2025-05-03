def aplicar_funcion_y_filtrar(lista, valor_umbral):
    return list(filter(lambda x: x > valor_umbral, map(lambda y: y**2, lista)))