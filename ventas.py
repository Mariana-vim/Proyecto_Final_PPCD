"""
Programacion para Ciencia de Datos - 03/junio/2025
Proyecto final: Analisis de datos en una Tienda de Conveniencia

Este script realiza un analisis de distintos aspectos de los datos de una tienda de conveniencia,
incluyendo exploracion, preprocesamiento, visualizacion y analisis de patrones de ventas.

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

def cargar_datos():
    datos_df = pd.read_csv('./convenience_store.csv')
    return datos_df
 

def explorar_datos(datos_df):

    print("\nA continuacion, se presentan las caracteristicas del dataset:\n")
    print(f"Forma del dataset: {datos_df.shape}")
    print(f"\nColumnas: {list(datos_df.columns)}")
    print(f"\nTipos de datos:")
    print(datos_df.dtypes)
    
    print(f"\nEstadisticas descriptivas:\n")
    print(datos_df.describe())

def preprocesar_datos(datos_df):
    
    # Se crea copia para no modificar los datos originales
    datos_procesados_df = datos_df.copy()
    
    # Se convierte la columna de fecha a un tipo de dato fecha y hora (datetime) usando pandas
    datos_procesados_df['Date'] = pd.to_datetime(datos_procesados_df['Date'])
    datos_procesados_df['Ano'] = datos_procesados_df['Date'].dt.year
    datos_procesados_df['Mes'] = datos_procesados_df['Date'].dt.month
    datos_procesados_df['Dia_Semana'] = datos_procesados_df['Date'].dt.dayofweek
    datos_procesados_df['Nombre_Dia'] = datos_procesados_df['Date'].dt.day_name()
    
    # Convertir tipos de datos numéricos
    datos_procesados_df['Total_Items'] = pd.to_numeric(datos_procesados_df['Total_Items'], errors='coerce')
    datos_procesados_df['Total_Cost'] = pd.to_numeric(datos_procesados_df['Total_Cost'], errors='coerce')
    
    # Remover caracteres especiales de productos
    datos_procesados_df['Product'] = datos_procesados_df['Product'].str.replace("'", "").str.replace("[", "").str.replace("]", "")
    
    # Crear categorias de momento del dia
    def momentos_dia(hora_str):
        try:
            hora = datetime.strptime(hora_str, '%H:%M:%S').hour
            if 6 <= hora < 12:
                return 'Mañana'
            elif 12 <= hora < 18:
                return 'Tarde'
            elif 18 <= hora < 24:
                return 'Noche'
            else:
                return 'Madrugada'
        except:
            return 'tiempo desconocido'
    
    datos_procesados_df['Periodo_Dia'] = datos_procesados_df['time_of_day'].apply(momentos_dia)
    
    print(f"\nDatos preprocesados, cuya forma final es: {datos_procesados_df.shape}")
    return datos_procesados_df

def analizar_patrones_venta(datos_df):
    print("\nA continuacion se presenta un analisis de patrones de venta:\n")
    
    resultados = {}
    
    # 1. Analisis de metodos de pago utilizados
    metodos_pago = datos_df['Payment_Method'].value_counts()
    resultados['metodos_pago'] = metodos_pago
    
    # 2. Analisis por temporada
    ventas_temporada = datos_df.groupby('Season')['Total_Cost'].agg(['sum', 'mean', 'count'])
    resultados['ventas_temporada'] = ventas_temporada
    
    # 3. Analisis por categoria de cliente
    ventas_categoria = datos_df.groupby('Customer_Category')['Total_Cost'].agg(['sum', 'mean', 'count'])
    resultados['ventas_categoria'] = ventas_categoria
    
    # 4. Analisis por periodo del día
    ventas_periodo = datos_df.groupby('Periodo_Dia')['Total_Cost'].agg(['sum', 'mean', 'count'])
    resultados['ventas_periodo'] = ventas_periodo
    
    # 5. Analisis de productos mas vendidos
    productos_individuales = []
    for productos in datos_df['Product'].str.split(', '):
        if isinstance(productos, list):
            productos_individuales.extend(productos)
    
    productos_serie = pd.Series(productos_individuales)
    productos_top = productos_serie.value_counts().head(10)
    resultados['productos_top'] = productos_top
    
    return resultados

def visualizaciones(datos_df, resultados):
    
    sns.set_style("whitegrid")
    
    # 1. Grafica de métodos de pago
    plt.figure(figsize=(10, 6))
    metodos_pago = resultados['metodos_pago']
    colores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    plt.subplot(2, 2, 1)
    barras = plt.bar(metodos_pago.index, metodos_pago.values, color=colores[:len(metodos_pago)])
    plt.title('Distribucion de metodos de pago', fontsize=14, fontweight='bold')
    plt.xlabel('Metodo de pago')
    plt.ylabel('Numero de transacciones')
    plt.xticks(rotation=45)
    
    # Agregar valores en las barras
    for barra in barras:
        altura = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2., altura,
                f'{int(altura)}', ha='center', va='bottom')
    
    # 2. Grafica de ventas por temporada
    plt.subplot(2, 2, 2)
    ventas_temporada = resultados['ventas_temporada']
    plt.bar(ventas_temporada.index, ventas_temporada['sum'], 
            color=['#D8A47F', '#A8D5BA', '#FFD93D', '#A0C4FF'])  # Fall, Spring, Summer, Winter
    plt.title('Ventas totales por temporada', fontsize=14, fontweight='bold')
    plt.xlabel('Temporada')
    plt.ylabel('Ventas totales ($)')
    plt.xticks(rotation=45)


    
    # 3. Grafica de categorias de clientes
    plt.subplot(2, 2, 3)
    ventas_categoria = resultados['ventas_categoria']
    plt.pie(ventas_categoria['count'], labels=ventas_categoria.index, autopct='%1.1f%%',
            colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC'])
    plt.title('Distribucion por categoria de cliente', fontsize=14, fontweight='bold')
    
    # 4. Grafica de productos mas vendidos
    plt.subplot(2, 2, 4)
    productos_top = resultados['productos_top'].head(8)
    colores_productos = ['#FFADAD', '#FFD6A5', '#FDFFB6', '#CAFFBF', 
                        '#9BF6FF', '#A0C4FF', '#BDB2FF', '#FFC6FF']

    plt.barh(range(len(productos_top)), productos_top.values, color=colores_productos)
    plt.yticks(range(len(productos_top)), productos_top.index)
    plt.title('Productos mas vendidos (Top 8)', fontsize=14, fontweight='bold')
    plt.xlabel('Numero de ventas')

    
    # Ajustar diseño y guardar
    plt.tight_layout()
    plt.savefig('results/figures/analisis_completo_tienda.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Graficas a parte
    individuales_graficas(datos_df, resultados)

def individuales_graficas(datos_df, resultados):

    # Grafica de tendencias por periodo del dia
    plt.figure(figsize=(12, 6))
    ventas_periodo = resultados['ventas_periodo']
    
    plt.subplot(1, 2, 1)
    plt.plot(ventas_periodo.index, ventas_periodo['mean'], marker='o', linewidth=3, markersize=8, color='coral')
    plt.title('Venta promedio por periodo del dia', fontsize=14, fontweight='bold')
    plt.xlabel('Periodo del dia')
    plt.ylabel('Venta promedio ($)')
    plt.grid(True, alpha=0.3)
    
    # Histograma de costos totales
    plt.subplot(1, 2, 2)
    plt.hist(datos_df['Total_Cost'], bins=30, alpha=0.7, color='lightpink', edgecolor='black')
    plt.title('Distribucion de costos totales', fontsize=14, fontweight='bold')
    plt.xlabel('Costo Total ($)')
    plt.ylabel('Frecuencia')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/analisis_detallado_tienda.png', dpi=300, bbox_inches='tight')
    plt.show()

def resumen_estadisticas(datos_df, resultados):

    resumen_estadisticas = {
        'Total_transacciones': [len(datos_df)],
        'Venta_total': [datos_df['Total_Cost'].sum()],
        'Venta_promedio': [datos_df['Total_Cost'].mean()],
        'Venta_mediana': [datos_df['Total_Cost'].median()],
        'Items_promedio_x_transaccion': [datos_df['Total_Items'].mean()],
        'Metodo_pago_mas_usado': [resultados['metodos_pago'].index[0]],
        'Temporada_mayor_venta': [resultados['ventas_temporada']['sum'].idxmax()],
        'Categoria_cliente_principal': [resultados['ventas_categoria']['count'].idxmax()],
        'Producto_mas_vendido': [resultados['productos_top'].index[0]]
    }
    
    resumen_df = pd.DataFrame(resumen_estadisticas)
    resumen_df.to_csv('results/tables/estadisticas_resumen.csv', index=False)
    
    print(f"• Total de transacciones analizadas: {len(datos_df):,}")
    print(f"• Venta total: ${datos_df['Total_Cost'].sum():,.2f}")
    print(f"• Venta promedio por transaccion: ${datos_df['Total_Cost'].mean():.2f}")
    print(f"• Método de pago mas usado: {resultados['metodos_pago'].index[0]}")
    print(f"• Temporada con mayores ventas: {resultados['ventas_temporada']['sum'].idxmax()}")
    
    return resumen_df

def analizar_promociones(datos_df):
    print("\nAnalisis de promociones y estrategias de marketing:\n")
    
    # Analisis de promociones existentes
    promociones = datos_df['Promotion'].value_counts()
    print("Promociones utilizadas:")
    for promo, count in promociones.items():
        print(f"  {promo}: {count} transacciones")
    
    # Ventas promedio con y sin promociones
    ventas_con_promo = datos_df[datos_df['Promotion'] != 'None']['Total_Cost'].mean()
    ventas_sin_promo = datos_df[datos_df['Promotion'] == 'None']['Total_Cost'].mean()
    
    print(f"\nVenta promedio con promociones: ${ventas_con_promo:.2f}")
    print(f"Venta promedio sin promociones: ${ventas_sin_promo:.2f}")
    print(f"Incremento por promociones: {((ventas_con_promo - ventas_sin_promo) / ventas_sin_promo * 100):.1f}%")
    
    # Analisis de membresias
    miembros = datos_df['Member'].value_counts()
    venta_miembros = datos_df[datos_df['Member'] == 'Yes']['Total_Cost'].mean()
    venta_no_miembros = datos_df[datos_df['Member'] == 'No']['Total_Cost'].mean()
    
    print(f"\nAnalisis de membresias:")
    print(f"Miembros: {miembros.get('Yes', 0)}, No miembros: {miembros.get('No', 0)}")
    print(f"Venta promedio miembros: ${venta_miembros:.2f}")
    print(f"Venta promedio no miembros: ${venta_no_miembros:.2f}")
    
    return {
        'promociones': promociones,
        'ventas_con_promo': ventas_con_promo,
        'ventas_sin_promo': ventas_sin_promo,
        'miembros': miembros,
        'venta_miembros': venta_miembros,
        'venta_no_miembros': venta_no_miembros
    }

def analizar_ciudades(datos_df):
    print("\nAnalisis de ventas por ciudad:\n")
    
    # Ventas por ciudad
    ventas_ciudad = datos_df.groupby('City')['Total_Cost'].agg(['sum', 'mean', 'count']).round(2)
    ventas_ciudad = ventas_ciudad.sort_values('sum', ascending=False)
    
    print("Top 5 ciudades por ventas totales:")
    for i, (ciudad, row) in enumerate(ventas_ciudad.head().iterrows(), 1):
        print(f"{i}. {ciudad}: ${row['sum']:.2f} ({row['count']} transacciones)")
    
    return ventas_ciudad

def graficas_marketing(datos_df, resultados_promo, ventas_ciudad):
    plt.figure(figsize=(15, 10))
    
    # 1. Impacto de promociones en ventas
    plt.subplot(2, 3, 1)
    categorias_promo = ['Con Promocion', 'Sin Promocion']
    valores_promo = [resultados_promo['ventas_con_promo'], resultados_promo['ventas_sin_promo']]
    colores_promo = ['#FF6B6B', '#4ECDC4']
    
    barras_promo = plt.bar(categorias_promo, valores_promo, color=colores_promo)
    plt.title('Impacto de promociones en ventas', fontweight='bold')
    plt.ylabel('Venta promedio ($)')
    
    for barra in barras_promo:
        altura = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2., altura,
                f'${altura:.2f}', ha='center', va='bottom')
    
    # 2. Distribucion de tipos de promociones
    plt.subplot(2, 3, 2)
    promociones = resultados_promo['promociones']
    plt.pie(promociones.values, labels=promociones.index, autopct='%1.1f%%',
            colors=['#FFB6C1', '#98FB98', '#87CEEB', '#DDA0DD'])
    plt.title('Distribucion de Promociones', fontweight='bold')
    
    # 3. Comparacion miembros vs no miembros
    plt.subplot(2, 3, 3)
    categorias_member = ['Miembros', 'No Miembros']
    valores_member = [resultados_promo['venta_miembros'], resultados_promo['venta_no_miembros']]
    colores_member = ['#FFD700', '#C0C0C0']
    
    barras_member = plt.bar(categorias_member, valores_member, color=colores_member)
    plt.title('Ventas: Miembros vs No Miembros', fontweight='bold')
    plt.ylabel('Venta promedio ($)')
    
    for barra in barras_member:
        altura = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2., altura,
                f'${altura:.2f}', ha='center', va='bottom')
    
    # 4. Top 10 ciudades por ventas totales
    plt.subplot(2, 3, 4)
    top_ciudades = ventas_ciudad.head(10)
    plt.barh(range(len(top_ciudades)), top_ciudades['sum'], color='lightcoral')
    plt.yticks(range(len(top_ciudades)), top_ciudades.index)
    plt.title('Top 10 ciudades por ventas', fontweight='bold')
    plt.xlabel('Ventas totales ($)')
    
    # 5. Relacion entre items y costo total
    plt.subplot(2, 3, 5)
    plt.scatter(datos_df['Total_Items'], datos_df['Total_Cost'], alpha=0.6, color='mediumseagreen')
    plt.title('Relacion items vs Costo total', fontweight='bold')
    plt.xlabel('Total de items')
    plt.ylabel('Costo total ($)')
    plt.grid(True, alpha=0.3)
    
    # 6. Heatmap de ventas por dia de semana y periodo
    plt.subplot(2, 3, 6)
    heatmap_data = datos_df.groupby(['Nombre_Dia', 'Periodo_Dia'])['Total_Cost'].mean().unstack()
    dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(dias_orden)
    
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Venta Promedio ($)'})
    plt.title('Ventas por dia y periodo', fontweight='bold')
    plt.xlabel('Periodo del dia')
    plt.ylabel('Dia de la semana')
    
    plt.tight_layout()
    plt.savefig('results/figures/analisis_marketing.png', dpi=300, bbox_inches='tight')
    plt.show()

def estrategias_recomendadas(datos_df, resultados_analisis, resultados_promo):
    print("\nEstrategias de marketing recomendadas:\n")
    
    # 1. Basado en productos mas vendidos
    producto_top = resultados_analisis['productos_top'].index[0]
    print(f"1. Promocionar {producto_top} como producto estrella (mas vendido)")
    
    # 2. Basado en temporadas
    temporada_top = resultados_analisis['ventas_temporada']['sum'].idxmax()
    print(f"2. Intensificar marketing en {temporada_top} (temporada de mayores ventas)")
    
    # 3. Basado en categorias de clientes
    categoria_top = resultados_analisis['ventas_categoria']['count'].idxmax()
    print(f"3. Dirigir campanas principales a {categoria_top} (segmento mas activo)")
    
    # 4. Basado en metodos de pago
    pago_top = resultados_analisis['metodos_pago'].index[0]
    print(f"4. Ofrecer incentivos por uso de {pago_top} (metodo preferido)")
    
    # 5. Basado en promociones
    if resultados_promo['ventas_con_promo'] > resultados_promo['ventas_sin_promo']:
        print("5. Incrementar promociones activas (generan mayor ticket promedio)")
    
    # 6. Basado en membresias
    if resultados_promo['venta_miembros'] > resultados_promo['venta_no_miembros']:
        print("6. Impulsar programa de membresias (miembros gastan mas)")
    
    print("\nOportunidades de crecimiento:")
    
    # Periodo con menor actividad
    periodo_bajo = resultados_analisis['ventas_periodo']['count'].idxmin()
    print(f"- Desarrollar promociones especiales para {periodo_bajo}")
    
    # Temporada con menor actividad
    temporada_baja = resultados_analisis['ventas_temporada']['sum'].idxmin()
    print(f"- Crear campanas estacionales para {temporada_baja}")
    
    # Productos menos vendidos que podrian tener potencial
    productos_menos_vendidos = resultados_analisis['productos_top'].tail(3)
    print("- Considerar descontinuar o promocionar agresivamente:")
    for producto in productos_menos_vendidos.index:
        print(f"  * {producto}")

def main():
    
    datos_df = cargar_datos()
    if datos_df is None:
        return
    
    explorar_datos(datos_df)
    
    datos_procesados_df = preprocesar_datos(datos_df)
    
    resultados_analisis = analizar_patrones_venta(datos_procesados_df)
    
    resultados_promo = analizar_promociones(datos_procesados_df)

    ventas_ciudad = analizar_ciudades(datos_procesados_df)
    
    visualizaciones(datos_procesados_df, resultados_analisis)
    
    graficas_marketing(datos_procesados_df, resultados_promo, ventas_ciudad)
    
    resumen_df = resumen_estadisticas(datos_procesados_df, resultados_analisis)

    estrategias_recomendadas(datos_procesados_df, resultados_analisis, resultados_promo)
    
if __name__ == "__main__":
    main()
