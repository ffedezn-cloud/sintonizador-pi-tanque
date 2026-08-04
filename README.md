# Sintonizador de Controlador PI para Tanque Calefaccionado

---

## Que hace este sintonizador

Sintonizar un controlar PI de nivel usando el método basado en modelos con valores recomendados o heurísticos como punto de partida. El usuario puede variar valores y decidir cuáles son los correctos con ayuda de gráficas interactivas, además tendrá un asistente que permite capacitarlo en este método de sintonía.

Permite ajustar:

- Ganancia proporcional (Kp)
- Efecto integral (con taui)
- Bias 

> Documentacion tecnica: [modelo_conceptual.pdf](assets/docs/modelo_conceptual.pdf)

---

## Modelo de espacio de estados 

El primer paso consiste en modelar el sistema completo, es decir, el tanque calefaccionado junto con el controlador PI acoplado. Este modelo permite realizar simulaciones sucesivas y ajustar los parametros del controlador mediante prueba y error. 
La estrategia de implementacion fue la siguiente:

- Implementar en Octave el modelo de espacio de estados
- Pasar al lenguaje Python la aplicacion
- Desplegar la aplicacion en Streamlit a traves de GitHub
- Usar IDEs: Geany y VSCodium, segun conveniencia, con IA generativa para detectar errores de codigo, para identacion automatica y sugerencias para mejorar experiencia frontend

---

## Tecnologias Utilizadas

- SO: AntiX Linux
- IDE: Geany / Geany Copilot - VSCodium / API DeepSeek

| Tecnologia | Proposito |
|------------|-----------|
| Octave | Modelado inicial y validacion |
| Python 3.8+ | Lenguaje necesario para desplegar en Streamlit |
| Streamlit | Interfaz web interactiva |
| Plotly | Graficas interactivas |
| SciPy | Resolucion de ecuaciones diferenciales |
| NumPy | Operaciones numericas |

---

## Como usar el sintonizador

### Opcion 1: En linea (recomendado)

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://sintonizador-pi-tanque.streamlit.app)

### Opcion 2: Localmente

Clonar el repositorio:

    git clone https://github.com/ffedezn-cloud/sintonizador-pi-tanque.git
    cd sintonizador-pi-tanque

Instalar dependencias:

    pip install -r requirements.txt

Ejecutar la aplicacion:

    streamlit run app.py

---

## Bibliografia

- Tarifa, E. (2025). Apuntes Simulacion y Optimizacion de Procesos. UNJu - FI.
- Ingham, J. (1994). Chemical Engineering Dynamics. Editorial VCH.
- Documentacion de Streamlit: https://docs.streamlit.io
- Documentacion de SciPy: https://docs.scipy.org

---

## Creditos

| Rol | Nombre |
|-----|--------|
| Autor | Federico Franco |
| Carrera | Ingenieria Quimica |
| Ano | 2026 |

---

## Licencia

Distribuido bajo licencia MIT. Ver el archivo LICENSE para mas informacion.

---

## Contacto

Federico Franco
ffede.zn@gmail.com

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/fede-franco-70a301418/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/ffedezn-cloud)

---

