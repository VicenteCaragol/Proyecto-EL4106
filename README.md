# Proyecto-EL4106

Proyecto 12. Clasificación de acciones en base a lecturas de electromiogramas.

Integrantes: Vicente Caragol, Christofer Cid

Los archivos .mat son los datos.

export_dataframe.csv es el estado actual del feature extraction de todo los datos. Para ver cómo se generó éste, revisar feature_ext.py

Clasificadores_sEMG.ipynb se utiliza para clasificar con el formato csv anterior, de aquí se obtuvieron las métricas de los clasificadores.

Para trabajar con la red convolucional se usaron 3 archivos: extractconv, Convnet-1 y Clasificadores_sEMG-2. Se usó la librería Pickle que guarda datos en archivos para acceder a ellos fácilmente desde otros programas.

Primeramente se usa extractconv.ipynb, una versión simplificada de feature_ext.py, para obtener los datos a usar a partir de los archivos .mat. Estos datos se guardan en un archivo .pickle 'Dicc.pickle'.

Posteriormente en Convnet-1.ipynb se carga el archivo 'Dicc.pickle' para usarlos en la red convolucional que se construye en el mismo archivo. En este programa se guardan los archivos 'output.pickle' correspondiente a la salida de la capa de características de la red, y el archivo 'Y.pickle', correspondiente a las clases de los datos usados.

Finalmente, Clasificadores_sEMG-2.ipynb corresponde a una modificación de Clasificadores_sEMG.ipynb donde los datos de entrada que se usan en los clasificadores corresponden al archivo 'output.pickle', es decir, los archivos extraídos de la red convolucional.
