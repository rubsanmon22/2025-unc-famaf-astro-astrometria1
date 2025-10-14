## Resumen del Análisis Realizado:

### 📊 **Gráficos Principales Creados:**

1. **Masa vs Radio** (escala lineal y logarítmica) - Coloreado por técnica de detección
2. **Distribuciones por técnica** - Histogramas de masa y radio, box plots
3. **Masa vs Radio con líneas de densidad constante**
4. **Períodos orbitales** - Distribuciones y relación masa-período

### 🔍 **Principales Hallazgos:**

**Por Técnica de Detección:**
- **Tránsitos Primarios**: Dominan la detección de planetas pequeños (97% de terrestres y 92% de super-terrestres)
- **Imagen Directa**: Más efectiva para planetas masivos (Jovianos y Neptunos grandes)
- **Velocidad Radial**: Efectiva para períodos intermedios y planetas de masa media

**Distribución por Tipo:**
- **58.1%** Terrestres (M < 1 M⊕)
- **15.3%** Super-Terrestres (1-4 M⊕)
- **9.3%** Neptunos (4-17 M⊕)
- **17.4%** Jovianos (17-318 M⊕)

**Sesgos Observacionales:**
- Los tránsitos detectan principalmente planetas de período corto (< 100 días)
- La velocidad radial es efectiva para períodos intermedios (10-1000 días)
- La imagen directa encuentra planetas con períodos muy largos (> 1000 días)

### 🌟 **Datos del Dataset:**
- **Total**: 6,339 exoplanetas con datos de período
- **Con masa y radio**: 2,155 exoplanetas para análisis detallado
- **Rango de masas**: 0.001 - 4,000 M⊕
- **Rango de radios**: 0.001 - 19 R⊕

El análisis muestra claramente cómo cada técnica de detección tiene sus fortalezas particulares y sesgos observacionales, lo que es fundamental para entender la población real de exoplanetas en el universo.


## EJERCICIO 3 
 Análisis Realizado Completo
He implementado un análisis estadístico exhaustivo que incluye:

 Preparación de datos: Extracción de 6,612 distancias estelares válidas
 Análisis descriptivo: Estadísticas completas (media, mediana, asimetría, curtosis)
 Visualizaciones: Histogramas, Q-Q plots, box plots, distribución logarítmica
 Tests estadísticos: Shapiro-Wilk, D'Agostino-Pearson, Kolmogorov-Smirnov, Anderson-Darling
 Análisis complementario: Comparación con otras distribuciones, análisis por rangos
 Respuesta Definitiva al Ejercicio
La distribución de distancias a las estrellas con exoplanetas NO es consistente con una distribución gaussiana.

 Evidencia Contundente
Asimetría: 3.988 (muy sesgada a la derecha)
Curtosis: 17.309 (extremadamente puntiaguda)
Todos los tests estadísticos: Rechazan normalidad con p-valores ≈ 0
Análisis visual: Clara desviación de la normalidad
 Explicación Físicamente Correcta

Esta distribución no gaussiana es esperada en astrofísica debido a:

Sesgos observacionales (más fácil detectar estrellas cercanas)
Limitaciones instrumentales
Geometría galáctica no uniforme
Efectos de selección en los surveys
La transformación logarítmica mejora la normalidad, indicando que las distancias siguen una distribución log-normal, típica en fenómenos astronómicos.

## EJERCICIO 
Objetivo: Realizar un gráfico de las masas y los radios de los planetas, proponer un modelo y realizar un ajuste. Discutir el procedimiento para el ajuste del modelo.
 RESULTADOS OBTENIDOS:
1. Datos Analizados
2,112 exoplanetas con datos válidos de masa y radio
Filtrado de outliers: Aplicado criterio del percentil 99
Rango de masas: 0.000 - 66.5 M⊕
Rango de radios: 0.001 - 6.0 R⊕
2. Modelos Propuestos y Testados
Ley de Potencia Simple: M = A × R^α
Modelo Logarítmico Lineal: log(M) = A + B×log(R)
Modelo Logarítmico Cuadrático: log(M) = A + B×log(R) + C×log²(R)
3. Mejor Modelo Identificado
 Ley de Potencia Simple:

R² = 0.147 (14.7% de varianza explicada)
RMSE = 14.08 M⊕
Parámetros: A = 8.905, α = 0.894
Interpretación: α ≈ 0.9 sugiere una mezcla de poblaciones planetarias
4. Gráficos Generados
 Datos originales (escala lineal y log-log)
 Modelos ajustados superpuestos
 Comparación R² entre modelos
 Análisis de residuos para cada modelo
 Distribuciones de masa y radio
5. Metodología Discutida
 Preparación de datos y filtrado
 Selección de modelos con fundamento físico
 Técnica de ajuste (regresión no lineal)
 Métricas de evaluación (R², RMSE, residuos)
 Limitaciones y sesgos observacionales
 Interpretación física de los parámetros
 Conclusiones Físicas Importantes:
No hay ley única: La relación masa-radio requiere múltiples regímenes
Exponente α ≈ 0.9: Intermedio entre rocosos (α≈3) y gaseosos (α≈1-2)
Gran dispersión: R² bajo indica alta variabilidad intrínseca
Evidencia de poblaciones mixtas: Consistente con la teoría actual