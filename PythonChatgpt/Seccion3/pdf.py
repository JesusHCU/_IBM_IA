import pandas as pd

# Definir el horario
horario_data = {
    "Lunes a Viernes": [
        ("6:30 - 7:00", "Ejercicio", "30 minutos"),
        ("7:00 - 8:00", "Estudio", "1 hora"),
        ("8:00 - 9:00", "Desayuno", "1 hora"),
        ("9:00 - 10:30", "Tareas del hogar", "1.5 horas"),
        ("10:30 - 12:00", "Estudio", "1.5 horas"),
        ("12:00 - 12:30", "Descanso", "30 minutos"),
        ("12:30 - 13:30", "Estudio", "1 hora"),
        ("13:30 - 14:00", "Preparar comida o Sacar a los perros", "30 minutos"),
        ("14:00 - 15:30", "Almuerzo familiar", "1.5 horas"),
        ("15:30 - 16:30", "Estudio", "1 hora"),
        ("16:30 - 17:30", "Descanso", "1 hora"),
        ("17:30 - 18:00", "Tiempo con tu hijo (jugar juntos)", "30 minutos"),
        ("18:00 - 19:00", "Tiempo con tu hijo (deberes)", "1 hora"),
        ("19:00 - 21:00", "Estudio", "2 horas"),
        ("21:00 - 21:30", "Sacar a los perros", "30 minutos"),
        ("21:30 - 22:30", "Preparar y Cena en familia", "1 hora"),
        ("22:30 - 00:30", "Descanso", "2 horas"),
    ],
    "Sábado y Domingo": [
        ("8:00 - 9:00", "Ejercicio físico", "1 hora"),
        ("9:00 - 10:00", "Estudio", "1 hora"),
        ("10:00 - 11:30", "Tareas del hogar", "1.5 horas"),
        ("11:30 - 13:00", "Descanso", "1.5 horas"),
        ("13:30 - 14:00", "Preparar y almorzar en familia", "30 minutos"),
        ("14:00 - 15:30", "Tiempo con tu hijo", "1.5 horas"),
        ("15:30 - 17:30", "Estudio", "2 horas"),
        ("17:30 - 18:00", "Descanso", "30 minutos"),
        ("18:00 - 20:00", "Estudio", "2 horas"),
        ("20:00 - 21:30", "Descanso", "1.5 horas"),
        ("21:30 - 22:30", "Cena", "1 hora"),
        ("22:30 - 01:00", "Descanso", "2.5 horas"),
    ],
}

# Crear un dataframe
df_lunes_a_viernes = pd.DataFrame(horario_data["Lunes a Viernes"], columns=["Hora", "Actividad", "Duración"])
df_sabado_domingo = pd.DataFrame(horario_data["Sábado y Domingo"], columns=["Hora", "Actividad", "Duración"])

# Crear el archivo Excel
excel_path = "Horario_semana.xlsx"
with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    df_lunes_a_viernes.to_excel(writer, sheet_name="Lunes a Viernes", index=False)
    df_sabado_domingo.to_excel(writer, sheet_name="Sábado y Domingo", index=False)

excel_path
