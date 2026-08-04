import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.integrate import odeint


# ===================================================================
#                    MODELO DE ESPACIO DE ESTADOS (Backend)
# ===================================================================

# ---------- PARÁMETROS FIJOS (Sistema SI) ----------
# Estos son los valores por defecto. En la interfaz se pueden modificar.
# Pero las funciones de simulación los usan como argumentos.
# No definimos constantes globales acá, las pasamos como parámetros.

# ---------- AEs (Ecuaciones Algebraicas/Constitutivas) ----------
def AEs(L, T, Ai, params, t=0):
    """
    Calcula las variables algebraicas (Y) a partir del estado X y parámetros.
    En Octave, esta función devolvía [A, F0, F, rho, Cp, T0, Q, Wa, e, Lsp, x, Ac].
    Aquí devolvemos un diccionario con las variables que necesitamos.
    """
    F0 = params['F0']; A = params['A']; Cv = params['Cv']; rho = params['rho']; g = params['g']
    Cp = params['Cp']; T0 = params['T0']; Tv = params['Tv']; Wa = params['Wa']; UAs = params['UAs']
    Ab = params['Ab']; Kp = params['Kp']; taui = params['taui']; Lsp = params['Lsp']

    # Cálculo del error según acción de control
    if params['accion'] == "Directa":
        e = L - Lsp
    else:
        e = Lsp - L
    
    # Señal del controlador PI
    Ac = Ab + Kp * (e + Ai / taui)
    x_frac = max(0.0, min(1.0, Ac))
    
    # Tipo de válvula: NC (0-1) o NA (1-0)
    if params.get('tipo_valvula', 'NC (0-1)') == "NA (1-0)":
        x_sat = 1.0 - x_frac
    else:
        x_sat = x_frac
    
    # Caudal de salida (modelo gravitatorio)
    L_segura = max(L, 0.001)
    F = Cv * x_sat * np.sqrt(rho * g * L_segura)
    
    # Calor transferido por el serpentín
    Q = UAs * (Tv - T)
    
    return {
        'A': A, 'F0': F0, 'F': F, 'rho': rho, 'Cp': Cp, 
        'T0': T0, 'Q': Q, 'Wa': Wa,
        'e': e, 'Lsp': Lsp, 'Ac': Ac, 'x': x_sat
    }

# ---------- ODEs (Ecuaciones Diferenciales) ----------
def ODEs(X, t, params):
    """
    Devuelve las derivadas de las variables de estado.
    """
    # Recuperar variables de estado
    L, T, Ai = X[0], X[1], X[2]
    L = max(L, 1e-4)
    
    # Calcular variables algebraicas
    Y = AEs(L, T, Ai, params, t)
    
    # Ecuaciones diferenciales
    dL = (Y['F0'] - Y['F']) / Y['A']
    dT = (Y['F0'] * Y['rho'] * Y['Cp'] * (Y['T0'] - T) + Y['Q'] + Y['Wa']) / (Y['A'] * L * Y['rho'] * Y['Cp'])
    dAi = Y['e']
    
    return [dL, dT, dAi]

# ---------- Inicialización ----------
def inicializar(params_por_defecto):
    """
    Inicializa la simulación con los parámetros dados.
    Devuelve las condiciones iniciales y las leyendas.
    """
    # Condiciones iniciales
    L0 = params_por_defecto['L0']
    T0 = params_por_defecto['T0_initial']
    Ai0 = params_por_defecto['Ai0']
    Xini = [L0, T0, Ai0]
    
    # Leyendas (para las gráficas)
    LX = ['L', 'T', 'Ai']   # Variables de estado
    LY = ['A', 'F0', 'F', 'rho', 'Cp', 'T0', 'Q', 'Wa', 'e', 'Lsp', 'x', 'Ac']  # Variables algebraicas
    
    return Xini, LX, LY

# ---------- Simulación ----------
def simulacion(tfin, dt, Xini, params):
    """
    Realiza la simulación dinámica.
    """
    # Vector de tiempo
    nts = int(np.ceil(tfin / dt)) + 1
    tpts = np.linspace(0, tfin, nts)
    
    # Resolver ODEs
    sol = odeint(ODEs, Xini, tpts, args=(params,))
    L, T, Ai = sol[:, 0], sol[:, 1], sol[:, 2]
    
    # Calcular variables dependientes (Y) en cada instante
    F = np.zeros_like(tpts)
    Q = np.zeros_like(tpts)
    e = np.zeros_like(tpts)
    Lsp = np.full_like(tpts, params['Lsp'])
    Ac = np.zeros_like(tpts)
    x_sat = np.zeros_like(tpts)
    
    for i in range(len(tpts)):
        Y = AEs(L[i], T[i], Ai[i], params, tpts[i])
        F[i] = Y['F']
        Q[i] = Y['Q']
        e[i] = Y['e']
        Lsp[i] = Y['Lsp']
        Ac[i] = Y['Ac']
        x_sat[i] = Y['x']
    
    return tpts, L, T, Ai, F, Q, e, Lsp, Ac, x_sat

# ---------- Validación de Acción ----------
def validar_accion(accion, tipo_valvula, modo):
    """
    Valida si la combinación acción-control + tipo de válvula es correcta.
    
    Servo (seguimiento): L debe seguir a Lsp
    Regulador (rechazo): L debe rechazar perturbaciones
    """
    if modo == "Servo":
        if accion == "Directa":
            if tipo_valvula == "NC (0-1)":
                return True, "Correcta: Directa + NC. Al aumentar Lsp, Ac disminuye, válvula NC se cierra, F disminuye, L aumenta (sigue a Lsp)."
            else:
                return False, "Incorrecta: Directa + NA. Al aumentar Lsp, Ac disminuye, válvula NA se abre, F aumenta, L disminuye. Pruebe Inversa."
        else:
            if tipo_valvula == "NA (1-0)":
                return True, "Correcta: Inversa + NA. Al aumentar Lsp, Ac aumenta, válvula NA se cierra, F disminuye, L aumenta (sigue a Lsp)."
            else:
                return False, "Incorrecta: Inversa + NC. Al aumentar Lsp, Ac aumenta, válvula NC se abre, F aumenta, L disminuye. Pruebe Directa."
    else:  # Regulador
        if accion == "Directa":
            if tipo_valvula == "NC (0-1)":
                return True, "Correcta: Directa + NC. Al aumentar L (perturbación), Ac aumenta, válvula NC se abre, F aumenta, L disminuye (rechaza perturbación)."
            else:
                return False, "Incorrecta: Directa + NA. Al aumentar L, Ac aumenta, válvula NA se cierra, F disminuye, L aumenta más. Pruebe Inversa."
        else:
            if tipo_valvula == "NA (1-0)":
                return True, "Correcta: Inversa + NA. Al aumentar L (perturbación), Ac disminuye, válvula NA se abre, F aumenta, L disminuye (rechaza perturbación)."
            else:
                return False, "Incorrecta: Inversa + NC. Al aumentar L, Ac disminuye, válvula NC se cierra, F disminuye, L aumenta más. Pruebe Directa."


# ===================================================================
#                    INTERFAZ DE USUARIO (Frontend)
# ===================================================================

# ---------- Configuración de la página ----------
st.set_page_config(
    page_title="Sintonizador PI - Tanque Calefaccionado", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- INYECCIÓN DE CSS PERSONALIZADO ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    footer {visibility: hidden;}
    /* #MainMenu visible para que el selector de temas funcione */
    
    .css-1d391kg, .css-1lcbmhc {
        background-color: #f5f6f7 !important;
        border-right: 1px solid #d0d0d5 !important;
        padding-top: 2rem !important;
    }
    
    .main .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 1rem !important;
        max-width: 1400px !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    h1 { font-size: 2rem !important; }
    h2 { font-size: 1.4rem !important; margin-top: 1.5rem !important; margin-bottom: 0.8rem !important; }
    h3 { font-size: 1.5rem !important; }
    
    h5 {
    font-weight: 400 !important;  /* Sin negrita */
    font-size: 0.95rem !important;
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important;
}
    
/* --- REGLAS DE ESPACIADO CORREGIDAS --- */
div[data-testid="stVerticalBlock"] {
    gap: 1rem !important; /* Le devolvemos respiro a los bloques (antes 0.2rem) */
}

.stNumberInput {
    padding-bottom: 0.2rem !important;
}
.stNumberInput label {
    font-size: 0.8rem !important;
    margin-bottom: 0.2rem !important; /* Quitamos el margen negativo que solapaba textos */
    padding-bottom: 0px !important;
}

h5 {
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important; /* Le damos espacio inferior a los títulos */
}

hr {
    margin: 1rem 0 !important;
}

.stMarkdown hr {
    margin: 1rem 0 !important;
}
/* --- FIN DE REGLAS DE ESPACIADO --- */
    
    .stButton > button {
        background-color: #e8e8ea !important;
        color: #1b1b32 !important;
        border: 1px solid #d0d0d5 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.4rem 0.8rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.01em !important;
        width: auto !important;
    }
    
    .stButton > button:hover {
        background-color: #1b1b32 !important;
        color: #ffffff !important;
        border-color: #1b1b32 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12) !important;
    }
    
    .stButton > button:active { transform: scale(0.98) !important; }
    
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 8px !important;
        border: 1px solid #d0d0d5 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div:focus {
        border-color: #1b1b32 !important;
        box-shadow: 0 0 0 2px rgba(27, 27, 50, 0.1) !important;
    }
    
    .streamlit-expanderHeader {
        font-weight: 500 !important;
        background-color: #f5f6f7 !important;
        border-radius: 8px !important;
        border: 1px solid #e8e8ee !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .streamlit-expanderHeader:hover { background-color: #e8e8ea !important; }
    
    div[data-testid="metric-container"] {
        border: 1px solid #d0d0d5 !important;
        border-radius: 12px !important;
        padding: 0.8rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.3s ease !important;
        background: transparent !important;
    }
    
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08) !important;
        transform: translateY(-2px) !important;
    }
    
    div[data-testid="metric-container"] > label {
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
    }
    
    .stAlert {
        border-radius: 10px !important;
        border-left: 4px solid #1b1b32 !important;
    }
    
    .stAlert > div {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }
    
    .stCaption, .caption {
        font-size: 0.75rem !important;
        font-weight: 400 !important;
    }
    
    @media (max-width: 768px) {
        .main .block-container { padding: 0.5rem 0.8rem !important; }
        .stButton > button { font-size: 0.8rem !important; padding: 0.4rem 0.6rem !important; }
        div[data-testid="metric-container"] { padding: 0.5rem !important; }
        div[data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    }
    
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #f5f6f7; }
    ::-webkit-scrollbar-thumb { background: #d0d0d5; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #b0b0b8; }
</style>
""", unsafe_allow_html=True)

# ---------- Título de la aplicación ----------
st.title("Sintonizador de Controlador PI de nivel para Tanque Calefaccionado")

# ---------- Imagen ----------
st.markdown(
    f'''
    <div style="text-align: center; margin: 10px 0;">
        <img src="https://raw.githubusercontent.com/ffedezn-cloud/sintonizador-pi-tanque/main/assets/images/diagrama_CL.png"
             alt="Esquema del tanque con control" 
             style="width: 60%; max-width: 450px; border: 1px solid #ddd; border-radius: 8px;">
        <p style="margin-top: 4px; font-size: 0.75rem; color: #888;">Esquema del tanque calefaccionado con control de nivel</p>
    </div>
    ''',
    unsafe_allow_html=True
)
st.markdown('<hr style="margin: 0.3rem 0;">', unsafe_allow_html=True)

# ---------- Barra lateral (Entrada de datos) ----------
with st.sidebar:
    # Información del desarrollador
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 10px 0;">
            <p style="font-size: 17px; font-weight: 600; margin-bottom: 2px;">Federico Franco</p>
            <p style="font-size: 16px; color: #888; margin-bottom: 2px;">Ingeniería Química</p>
            <a href="mailto:ffede.zn@gmail.com" style="font-size: 16px; color: #888; text-decoration: none;">
                ffede.zn@gmail.com
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.subheader("Datos Geométricos del Tanque")
    D = st.number_input("Diámetro del tanque D (m)", value=1.0, min_value=0.3, max_value=5.0, step=0.05)
    A = np.pi * (D/2)**2
    st.caption(f"Area calculada: {A:.4f} m²")
    
    L0 = st.number_input("Nivel inicial L0 (m)", value=1.0, min_value=0.0, max_value=5.0, step=0.05)
    L_max = st.number_input("Nivel máximo (rebalse) L_max (m)", value=2.0, min_value=0.5, max_value=10.0, step=0.1)
    
    st.subheader("Datos de Operación")
    F0 = st.number_input("Caudal de entrada F0 (m³/s)", value=0.002, min_value=0.0001, max_value=0.1, format="%.5f")
    
    st.subheader("Datos del Fluido")
    rho = st.number_input("Densidad del fluido ρ (kg/m³)", value=1000.0, min_value=500.0, max_value=2000.0, step=10.0)
    g = 9.81
    st.caption(f"Gravedad fija: g = {g} m/s²")
    
    st.subheader("Parámetros del sistema")
    t_final = st.slider("Tiempo de simulación (s)", 100, 5000, 2000, 100)

# ---------- Parámetros fijos ----------
params_fijos = {
    'A': A, 'F0': F0, 'rho': rho, 'g': g, 'L0': L0,
    'Cp': 4187, 'T0': 25, 'Tv': 132, 'Wa': 2000, 'T0_initial': 60,
    'Cv': 4.039e-5, 'UAs': 4.04e3, 'Ai0': 0
}

# ---------- Mostrar parámetros calculados en tarjetas ----------
st.subheader("Parámetros del Sistema")

col_a, col_b, col_c, col_d, col_e, col_f = st.columns(6)
with col_a:
    st.metric("Area del tanque A", f"{A:.4f} m²")
with col_b:
    st.metric("Cv (valvula)", f"{params_fijos['Cv']:.4e}")
with col_c:
    st.metric("Caudal de entrada F0", f"{F0:.4f} m³/s")
with col_d:
    st.metric("UA", f"{params_fijos['UAs']:.0f} W/°C")
with col_e:
    st.metric("Nivel inicial L0", f"{L0:.2f} m")
with col_f:
    st.metric("Temperatura inicial", f"60.0 °C")

st.markdown('<hr style="margin: 1rem 0;">', unsafe_allow_html=True)


# ===================================================================
#                    SECCIÓN 1: SINTONÍA INTERACTIVA
# ===================================================================

st.subheader("Sintonía Interactiva del Controlador PI de nivel")

# --- Selección de la Acción de Control ---
st.markdown("##### Selección de la Acción de Control")
col_acc1, col_acc2, col_acc3 = st.columns(3)
with col_acc1:
    accion = st.selectbox("Acción del controlador", ["", "Directa", "Inversa"])
with col_acc2:
    modo = st.selectbox("Modo de operación", ["Servo", "Regulador"])
with col_acc3:
    tipo_valvula_sel = st.selectbox("Tipo de válvula", ["NC (0-1)", "NA (1-0)"])

accion_seleccionada = accion != ""

if accion_seleccionada:
    es_correcta, mensaje = validar_accion(accion, tipo_valvula_sel, modo)
    if es_correcta:
        st.success(f"{mensaje}")
    else:
        st.error(f"{mensaje}")
else:
    st.info("Seleccione una acción para comenzar el proceso de sintonía.")

st.markdown('<hr style="margin: 0.8rem 0;">', unsafe_allow_html=True)

if accion_seleccionada:
    # ---------- Valores iniciales ----------
    if 'Ab' not in st.session_state:
        st.session_state.Ab = 0.0
    if 'Kp' not in st.session_state:
        st.session_state.Kp = 0.0
    if 'taui' not in st.session_state:
        st.session_state.taui = 1.0
    if 'Ai0' not in st.session_state:
        st.session_state.Ai0 = 0.5
    if 'Lsp_set' not in st.session_state:
        st.session_state.Lsp_set = 0.0

    # ============================================================
    #           FILA ÚNICA: Parámetros + Guía (lado a lado)
    # ============================================================
    col_param, col_guia = st.columns([2, 1.5], gap="medium")

    with col_param:
        st.markdown("##### Parámetros del controlador")
        
        # 5 columnas compactas para los parámetros
        c1, c2, c3, c4, c5 = st.columns(5, gap="small")
        
        with c1:
            st.session_state.Ab = st.number_input(
                "Bias Ab", 0.0, 1.0, st.session_state.Ab, 0.01, format="%.2f",
                key="ab_input"
            )
        with c2:
            st.session_state.Kp = st.number_input(
                "Kp", 0.0, 25.0, st.session_state.Kp, 0.1, format="%.1f",
                key="kp_input"
            )
        with c3:
            st.session_state.taui = st.number_input(
                "τi [s]", 1.0, 15000.0, st.session_state.taui, 10.0, format="%.0f",
                key="taui_input"
            )
        with c4:
            st.session_state.Ai0 = st.number_input(
                "Ai(0)", -5.0, 5.0, st.session_state.Ai0, 0.1, format="%.1f",
                key="ai0_input"
            )
        with c5:
            st.session_state.Lsp_set = st.number_input(
                "Lsp", 0.0, 5.0, st.session_state.Lsp_set, 0.05, format="%.2f",
                key="lsp_set_input"
            )

    with col_guia:
        st.markdown("##### Guía de sintonía")
        
        Ab = st.session_state.Ab
        Kp = st.session_state.Kp
        taui = st.session_state.taui
        Ai0 = st.session_state.Ai0
        Lsp_set = st.session_state.Lsp_set

        # Simulación para cálculos de la guía
        def Lsp_step(t):
            return Lsp_set if t >= 400.0 else 1.0

        def ODEs_step(X, t, params):
            L, T, Ai = X[0], X[1], X[2]
            L = max(L, 1e-4)
            params = params.copy()
            params['Lsp'] = Lsp_step(t)
            Y = AEs(L, T, Ai, params, t)
            dL = (Y['F0'] - Y['F']) / Y['A']
            dT = (Y['F0']*Y['rho']*Y['Cp']*(Y['T0']-T) + Y['Q'] + Y['Wa']) / (Y['A']*L*Y['rho']*Y['Cp'])
            dAi = Y['e']
            return [dL, dT, dAi]

        params = params_fijos.copy()
        params.update({
            'Ab': Ab,
            'Kp': Kp,
            'taui': max(taui, 1.0),
            'accion': accion,
            'tipo_valvula': tipo_valvula_sel,
            'Lsp': 1.0,
            'Ai0': Ai0
        })

        tpts = np.linspace(0, t_final, int(t_final / 10) + 1)
        X0 = [L0, 60.0, Ai0]
        sol = odeint(ODEs_step, X0, tpts, args=(params,))
        L = sol[:, 0]
        Ai = sol[:, 2]

        mask_pre = tpts < 380
        L_pre = L[mask_pre]
        offset_pre = float(np.mean(L_pre[-20:]) - 1.0) if len(L_pre) > 20 else 0.0
        ai_final = float(Ai[-1])

        kp_ok = 2.0 <= Kp <= 20.0
        taui_off = taui >= 14000
        taui_transicion = 300 < taui < 14000
        ai0_ok = abs(Ai0) < 0.05
        lsp_ok = Lsp_set > 0.05
        offset_ok = abs(offset_pre) < 0.025
        ai_ok = abs(ai_final) < 0.25

        # Mensajes de guía en una sola línea
        if taui_off:
            if not kp_ok:
                st.warning(
                    "**Paso 1:** Colocá Kp según heurística, entre 2 y 20.\n\n"
                   
                )
            elif not ai0_ok:
                st.warning(
                    "**Paso 2:** Inicializá el efecto integral en cero: Ai(0)=0.\n\n"
                    
                )
            elif not lsp_ok:
                st.warning(
                    "**Paso 3:** Fijá el Lsp al valor deseado.\n\n"
                   
                )
            elif not offset_ok:
                direccion = "subí" if offset_pre < 0 else "bajá"
                st.warning(
                    f"**Paso 4:** Ajustá el Bias (Ab) hasta que L = Lsp.\n\n"
                   
                )
            else:
                st.success(
                    "**Parte 1 completa:** Habilitar el efecto integral reduciendo **τi** hasta el rango recomendado (30 – 300 s).\n\n"
                  
                )
        
        elif taui_transicion:
            if not kp_ok or not ai0_ok or not lsp_ok or not offset_ok:
                st.warning(
                   
                    "Subí **τi a 15000** para anular la integral,\n"

                )
            else:
                st.info(
                    f"**Reduciendo τi para activar la integral**\n\n"
                    f"τi actual = **{taui:.0f} s**\n\n"
                    "Seguí reduciendo hasta el rango recomendado: **30 – 300 s**.\n\n"
                    f"Kp = **{Kp:.1f}** · Ab = **{Ab:.3f}** · Lsp = **{Lsp_set:.2f}**"
                )
        
        else:
            if not kp_ok or not ai0_ok or not lsp_ok or not offset_ok:
                st.warning(
                    
                    "Subí **τi al máximo (15000)** para anular el efecto integral\n"
                    "y completá el ajuste de **Reset Manual**."
                )
            elif abs(ai_final) > 0.5:
                if ai_final > 0:
                    st.warning(
                        f"**Ai = {ai_final:+.2f} > 0 → el Ab quedó bajo**\n\n"
                        "El efecto integral está compensando la falta de bias.\n\n"
                        "**Solución:** Subí **τi a 15000**, aumentá un poco **Ab**\n"
                        "y repetí el proceso de ajuste."
                    )
                else:
                    st.warning(
                        f" **Ai = {ai_final:+.2f} < 0 → el Ab quedó alto**\n\n"
                        "El efecto integral está compensando el exceso de bias.\n\n"
                        "**Solución:** Subí **τi a 15000**, bajá un poco **Ab**\n"
                        "y repetí el proceso de ajuste."
                    )
            elif not ai_ok:
                st.info(
                    f" **Ajuste fino** | Ai = **{ai_final:+.2f}** (cercano a 0)\n\n"
                    "Verificá en la gráfica que **Ai se mantenga cerca de cero**.\n\n"
                    f"τi = **{taui:.0f} s** · Kp = **{Kp:.1f}** · Ab = **{Ab:.3f}**"
                )
            else:
                st.success(
                    f" **Sintonía completa según el método basado en modelos**. Ver documentación para más detalles.\n\n"
                    
                    
                )

    st.markdown('<hr style="margin: 0.3rem 0;">', unsafe_allow_html=True)
    
    # ---------- GRÁFICA ÚNICA CON DOBLE EJE ----------
    st.markdown("##### Gráfica de la Simulación")
    
    # Re-ejecutar simulación para la gráfica
    def Lsp_step(t):
        return Lsp_set if t >= 400.0 else 1.0

    def ODEs_step(X, t, params):
        L, T, Ai = X[0], X[1], X[2]
        L = max(L, 1e-4)
        params = params.copy()
        params['Lsp'] = Lsp_step(t)
        Y = AEs(L, T, Ai, params, t)
        dL = (Y['F0'] - Y['F']) / Y['A']
        dT = (Y['F0']*Y['rho']*Y['Cp']*(Y['T0']-T) + Y['Q'] + Y['Wa']) / (Y['A']*L*Y['rho']*Y['Cp'])
        dAi = Y['e']
        return [dL, dT, dAi]

    params = params_fijos.copy()
    params.update({
        'Ab': Ab,
        'Kp': Kp,
        'taui': max(taui, 1.0),
        'accion': accion,
        'tipo_valvula': tipo_valvula_sel,
        'Lsp': 1.0,
        'Ai0': Ai0
    })

    tpts = np.linspace(0, t_final, int(t_final / 10) + 1)
    X0 = [L0, 60.0, Ai0]
    sol = odeint(ODEs_step, X0, tpts, args=(params,))
    L = sol[:, 0]
    Ai = sol[:, 2]
    Ac_plot = np.zeros_like(tpts)
    x_sat_plot = np.zeros_like(tpts)
    Lsp_plot = np.zeros_like(tpts)
    for i, t in enumerate(tpts):
        params['Lsp'] = Lsp_step(t)
        Y = AEs(L[i], 60.0, Ai[i], params, t)
        Ac_plot[i] = Y['Ac']
        x_sat_plot[i] = Y['x']
        Lsp_plot[i] = params['Lsp']

    # Detectar tema del navegador
    tema_oscuro = st.get_option("theme.base") == "dark"

    if tema_oscuro:
        bg_color = 'rgba(30,30,30,0.95)'
        text_color = 'white'
        grid_color = 'rgba(255,255,255,0.15)'
        legend_bg = 'rgba(0,0,0,0.6)'
        template = "plotly_dark"
        colors = {
            'primary': '#4dabf7',
            'success': '#51cf66',
            'danger': '#ff6b6b',
            'secondary': '#868e96',
            'warning': '#fcc419'
        }
    else:
        bg_color = 'white'
        text_color = 'black'
        grid_color = 'rgba(0,0,0,0.1)'
        legend_bg = 'rgba(255,255,255,0.8)'
        template = "plotly_white"
        colors = {
            'primary': '#1f77b4',
            'success': '#2ca02c',
            'danger': '#d62728',
            'secondary': '#7f7f7f',
            'warning': '#fcc419'
        }

    config_plotly = {
        'scrollZoom': False,
        'displayModeBar': True,
        'responsive': True
    }

    # Gráfica única con doble eje - MEJORADA PARA MÓVILES
    fig = go.Figure()
    
    # Eje izquierdo: Nivel
    fig.add_trace(go.Scatter(
        x=tpts, y=L, 
        mode='lines', 
        name='Nivel L(t)',
        line=dict(color=colors['primary'], width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=tpts, y=Lsp_plot, 
        mode='lines', 
        name='Lsp',
        line=dict(color=colors['danger'], width=2, dash='dash')
    ))
    
    # Línea horizontal de L_max
    fig.add_hline(
        y=L_max, 
        line=dict(color=colors['danger'], width=1.5, dash='dot')
    )
    
    fig.add_annotation(
        x=50, y=L_max+0.05, 
        text=f'L_max = {L_max} m', 
        showarrow=False,
        font=dict(color=text_color, size=9),
        bgcolor=legend_bg,
        bordercolor=colors['danger'],
        borderwidth=1
    )
    
    # Eje derecho: Ac, x, Ai
    fig.add_trace(go.Scatter(
        x=tpts, y=Ac_plot, 
        mode='lines', 
        name='Ac',
        line=dict(color=colors['warning'], width=2, dash='dot'),
        yaxis='y2'
    ))
    fig.add_trace(go.Scatter(
        x=tpts, y=x_sat_plot, 
        mode='lines', 
        name='x',
        line=dict(color=colors['success'], width=2),
        yaxis='y2'
    ))
    fig.add_trace(go.Scatter(
        x=tpts, y=Ai, 
        mode='lines', 
        name='Ai',
        line=dict(color=colors['secondary'], width=2, dash='dashdot'),
        yaxis='y2'
    ))
    
    # Líneas horizontales en el eje y2
    fig.add_shape(
        type='line',
        y0=0, y1=0,
        x0=0, x1=1,
        xref='paper',
        yref='y2',
        line=dict(color='gray', width=1, dash='dash')
    )
    fig.add_shape(
        type='line',
        y0=1, y1=1,
        x0=0, x1=1,
        xref='paper',
        yref='y2',
        line=dict(color='gray', width=1, dash='dash')
    )

    fig.update_layout(
        template=template,
        dragmode=False,
        title=dict(text='Nivel y Señales del Controlador', font=dict(color=text_color, size=13)),
        xaxis=dict(
            title='Tiempo (s)',
            title_font=dict(color=text_color, size=11),
            tickfont=dict(color=text_color, size=9),
            gridcolor=grid_color,
            showgrid=True,
            zeroline=True,
            zerolinecolor=grid_color,
            showline=True,
            linecolor=grid_color
        ),
        yaxis=dict(
            title='Nivel L (m)',
            title_font=dict(color=text_color, size=11),
            tickfont=dict(color=text_color, size=9),
            gridcolor=grid_color,
            showgrid=True,
            zeroline=True,
            zerolinecolor=grid_color,
            showline=True,
            linecolor=grid_color,
            range=[0, max(float(L_max)+0.15, float(np.max(L))+0.12)]
        ),
        yaxis2=dict(
            title='Ac, x, Ai',
            title_font=dict(color=text_color, size=11),
            tickfont=dict(color=text_color, size=9),
            overlaying='y',
            side='right',
            showgrid=False,
            range=[-3.5, 3.5]
        ),
        height=350,
        hovermode='x unified',
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color, size=9),
        legend=dict(
            font=dict(color=text_color, size=10),
            bgcolor=legend_bg,
            orientation='h',
            yanchor='top',
            y=-0.25,  # Leyenda debajo del eje X
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=10, r=40, t=40, b=40)  # Márgenes reducidos, derecho más grande para eje y2
    )
    st.plotly_chart(fig, use_container_width=True, config=config_plotly)

st.markdown('<hr style="margin: 1rem 0;">', unsafe_allow_html=True)

# ===================================================================
#                    SECCIÓN 2: ANÁLISIS DE OFFSET VARIABLE
# ===================================================================

st.subheader("Análisis de Offset Variable")

if not accion_seleccionada:
    st.warning("Seleccione una acción en la sección anterior para acceder al análisis de offset variable.")
else:
    # ---------- Inicializar lista de perturbaciones ----------
    if 'perturbaciones' not in st.session_state:
        st.session_state.perturbaciones = []
    if 'simulaciones_guardadas' not in st.session_state:
        st.session_state.simulaciones_guardadas = []

    # ============================================================
    #           DOS COLUMNAS: Izquierda (Agregar) y Derecha (Lista)
    # ============================================================
    col_agregar, col_lista = st.columns([1, 1.5], gap="medium")

    # ---------- COLUMNA IZQUIERDA: Agregar perturbación ----------
    with col_agregar:
        st.markdown("##### Agregar nueva perturbación")
        
        with st.form(key="form_perturbacion", clear_on_submit=True):
            nuevo_t = st.number_input(
                "Tiempo (s)", 
                min_value=0, 
                max_value=5000, 
                step=50, 
                value=300,
                placeholder="Ej: 300"
            )
            nuevo_lsp = st.number_input(
                "Lsp (m)", 
                min_value=0.1, 
                max_value=5.0, 
                step=0.05, 
                value=0.40,
                format="%.2f"
            )
            submit_button = st.form_submit_button("Agregar", use_container_width=True)
            
            if submit_button:
                nombre = f"t={nuevo_t}s, Lsp={nuevo_lsp:.2f}"
                st.session_state.perturbaciones.append((nuevo_t, nuevo_lsp, nombre))
                st.session_state.perturbaciones.sort(key=lambda x: x[0])
                st.session_state.simulaciones_guardadas = []
                st.rerun()

    # ---------- COLUMNA DERECHA: Lista de perturbaciones ----------
    with col_lista:
        st.markdown("##### Perturbaciones activas")
        
        if len(st.session_state.perturbaciones) == 0:
            st.info("No hay perturbaciones agregadas.")
        else:
            for i, (t, val, nombre) in enumerate(st.session_state.perturbaciones):
                r_col1, r_col2, r_col3 = st.columns([1.2, 1.5, 0.8])
                with r_col1:
                    st.write(f"{t:.2f}")
                with r_col2:
                    st.write(f"{val:.2f}")
                with r_col3:
                    if st.button("Eliminar", key=f"del_{i}", use_container_width=True):
                        st.session_state.perturbaciones.pop(i)
                        st.session_state.simulaciones_guardadas.pop(i)
                        st.rerun()

    st.markdown('<hr style="margin: 0.3rem 0;">', unsafe_allow_html=True)

    # ---------- GRÁFICAS EN PESTAÑAS ----------
    Ab_off = Ab if accion_seleccionada else 0.3
    Kp_off = Kp if accion_seleccionada else 5.0
    taui_off = taui if accion_seleccionada else 100
    accion_off = accion if accion_seleccionada else "Directa"
    tipo_valvula_off = tipo_valvula_sel if accion_seleccionada else "NC (0-1)"
    
    params_off = params_fijos.copy()
    params_off.update({
        'Ab': Ab_off,
        'Kp': Kp_off,
        'taui': taui_off,
        'accion': accion_off,
        'tipo_valvula': tipo_valvula_off
    })
    
    def simular_perturbacion(t_pert, lsp_pert, params_off, t_final, L0):
        def Lsp_escalon(t):
            return lsp_pert if t >= t_pert else 1.0
        
        def ODEs_off(X, t, params_off):
            L, T, Ai = X[0], X[1], X[2]
            L = max(L, 1e-4)
            params_off['Lsp'] = Lsp_escalon(t)
            Y = AEs(L, T, Ai, params_off, t)
            dL = (Y['F0'] - Y['F']) / Y['A']
            dT = (Y['F0'] * Y['rho'] * Y['Cp'] * (Y['T0'] - T) + Y['Q'] + Y['Wa']) / (Y['A'] * L * Y['rho'] * Y['Cp'])
            dAi = Y['e']
            return [dL, dT, dAi]
        
        tpts_off = np.linspace(0, t_final, 1000)
        X0_off = [L0, 60.0, 0.0]
        
        sol_off = odeint(ODEs_off, X0_off, tpts_off, args=(params_off,))
        L_off = sol_off[:, 0]
        Ai_off = sol_off[:, 2]
        
        Lsp_plot = np.zeros_like(tpts_off)
        Ac_plot = np.zeros_like(tpts_off)
        x_plot = np.zeros_like(tpts_off)
        for i, t in enumerate(tpts_off):
            params_off['Lsp'] = Lsp_escalon(t)
            Y = AEs(L_off[i], 60.0, Ai_off[i], params_off, t)
            Lsp_plot[i] = params_off['Lsp']
            Ac_plot[i] = Y['Ac']
            x_plot[i] = Y['x']
        
        return tpts_off, L_off, Lsp_plot, Ac_plot, x_plot
    
    if len(st.session_state.simulaciones_guardadas) != len(st.session_state.perturbaciones):
        st.session_state.simulaciones_guardadas = []
        for t_pert, lsp_pert, nombre in st.session_state.perturbaciones:
            tpts, L, Lsp, Ac, x = simular_perturbacion(t_pert, lsp_pert, params_off, t_final, L0)
            st.session_state.simulaciones_guardadas.append({
                'nombre': nombre,
                't_pert': t_pert,
                'lsp_pert': lsp_pert,
                'tpts': tpts,
                'L': L,
                'Lsp': Lsp,
                'Ac': Ac,
                'x': x
            })

    # Detectar tema del navegador
    tema_oscuro = st.get_option("theme.base") == "dark"

    if tema_oscuro:
        bg_color = 'rgba(30,30,30,0.95)'
        text_color = 'white'
        grid_color = 'rgba(255,255,255,0.15)'
        legend_bg = 'rgba(0,0,0,0.6)'
        template = "plotly_dark"
    else:
        bg_color = 'white'
        text_color = 'black'
        grid_color = 'rgba(0,0,0,0.1)'
        legend_bg = 'rgba(255,255,255,0.8)'
        template = "plotly_white"

    config_plotly = {
        'scrollZoom': False,
        'displayModeBar': True,
        'responsive': True
    }

    colores_off = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    tab1, tab2 = st.tabs(["Señal del controlador", "Nivel vs tiempo"])

    with tab1:
        with st.container(border=True):
            fig_off1 = go.Figure()
            if len(st.session_state.simulaciones_guardadas) > 0:
                for idx, sim in enumerate(st.session_state.simulaciones_guardadas):
                    color = colores_off[idx % len(colores_off)]
                    nombre = sim['nombre']
                    
                    fig_off1.add_trace(go.Scatter(
                        x=sim['tpts'], y=sim['Ac'],
                        mode='lines',
                        name=f"{nombre} - Ac",
                        line=dict(color=color, width=2, dash='solid')
                    ))
                    
                    fig_off1.add_trace(go.Scatter(
                        x=sim['tpts'], y=sim['x'],
                        mode='lines',
                        name=f"{nombre} - x",
                        line=dict(color=color, width=2, dash='dash')
                    ))
                
                fig_off1.add_hline(y=0, line=dict(color='gray', width=1, dash='dash'))
                fig_off1.add_hline(y=1, line=dict(color='gray', width=1, dash='dash'))
                
                fig_off1.update_layout(
                    template=template,
                    dragmode=False,
                    title=dict(text='Señal del controlador y apertura de válvula', font=dict(color=text_color, size=14)),
                    xaxis=dict(
                        title='Tiempo (s)',
                        title_font=dict(color=text_color, size=12),
                        tickfont=dict(color=text_color, size=10),
                        gridcolor=grid_color,
                        showgrid=True,
                        zeroline=True,
                        zerolinecolor=grid_color,
                        showline=True,
                        linecolor=grid_color
                    ),
                    yaxis=dict(
                        title='Señal',
                        title_font=dict(color=text_color, size=12),
                        tickfont=dict(color=text_color, size=10),
                        gridcolor=grid_color,
                        showgrid=True,
                        zeroline=True,
                        zerolinecolor=grid_color,
                        showline=True,
                        linecolor=grid_color,
                        range=[-0.15, 1.15]
                    ),
                    height=350,
                    hovermode='x unified',
                    plot_bgcolor=bg_color,
                    paper_bgcolor=bg_color,
                    font=dict(color=text_color, size=10),
                    legend=dict(
                        font=dict(color=text_color, size=8),
                        bgcolor=legend_bg,
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='center',
                        x=0.5
                    ),
                    margin=dict(l=40, r=20, t=35, b=35)
                )
            else:
                fig_off1.add_annotation(
                    text="Agregue perturbaciones para visualizar<br>la señal del controlador",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(color=text_color, size=14)
                )
                fig_off1.update_layout(
                    template=template,
                    title=dict(text='Señal del controlador y apertura de válvula', font=dict(color=text_color, size=14)),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    height=350,
                    plot_bgcolor=bg_color,
                    paper_bgcolor=bg_color,
                    font=dict(color=text_color, size=10),
                    margin=dict(l=40, r=20, t=35, b=35)
                )
            st.plotly_chart(fig_off1, use_container_width=True, config=config_plotly)

    with tab2:
        with st.container(border=True):
            fig_off2 = go.Figure()
            if len(st.session_state.simulaciones_guardadas) > 0:
                for idx, sim in enumerate(st.session_state.simulaciones_guardadas):
                    color = colores_off[idx % len(colores_off)]
                    nombre = sim['nombre']
                    
                    fig_off2.add_trace(go.Scatter(
                        x=sim['tpts'], y=sim['L'],
                        mode='lines',
                        name=f"{nombre} - L(t)",
                        line=dict(color=color, width=2)
                    ))
                    
                    fig_off2.add_trace(go.Scatter(
                        x=sim['tpts'], y=sim['Lsp'],
                        mode='lines',
                        name=f"{nombre} - Lsp",
                        line=dict(color=color, width=2, dash='dash')
                    ))
                
                fig_off2.add_hline(y=L_max, line=dict(color='#d62728', width=1.5, dash='dot'))
                fig_off2.add_annotation(
                    x=50, y=L_max+0.05,
                    text=f'L_max = {L_max} m',
                    showarrow=False,
                    font=dict(color=text_color, size=10),
                    bgcolor=legend_bg,
                    bordercolor='#d62728',
                    borderwidth=1
                )
                
                fig_off2.update_layout(
                    template=template,
                    title=dict(text='Nivel vs tiempo', font=dict(color=text_color, size=14)),
                    xaxis=dict(
                        title='Tiempo (s)',
                        title_font=dict(color=text_color, size=12),
                        tickfont=dict(color=text_color, size=10),
                        gridcolor=grid_color,
                        showgrid=True,
                        zeroline=True,
                        zerolinecolor=grid_color,
                        showline=True,
                        linecolor=grid_color
                    ),
                    yaxis=dict(
                        title='L (m)',
                        title_font=dict(color=text_color, size=12),
                        tickfont=dict(color=text_color, size=10),
                        gridcolor=grid_color,
                        showgrid=True,
                        zeroline=True,
                        zerolinecolor=grid_color,
                        showline=True,
                        linecolor=grid_color
                    ),
                    height=350,
                    hovermode='x unified',
                    plot_bgcolor=bg_color,
                    paper_bgcolor=bg_color,
                    font=dict(color=text_color, size=10),
                    legend=dict(
                        font=dict(color=text_color, size=8),
                        bgcolor=legend_bg,
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='center',
                        x=0.5
                    ),
                    margin=dict(l=40, r=20, t=35, b=35)
                )
                max_L = max([sim['L'].max() for sim in st.session_state.simulaciones_guardadas])
                fig_off2.update_yaxes(range=[0, max(2.5, max_L*1.1)])
            else:
                fig_off2.add_annotation(
                    text="Agregue perturbaciones para visualizar<br>el nivel del tanque",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(color=text_color, size=14)
                )
                fig_off2.update_layout(
                    template=template,
                    title=dict(text='Nivel vs tiempo', font=dict(color=text_color, size=14)),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    height=350,
                    plot_bgcolor=bg_color,
                    paper_bgcolor=bg_color,
                    font=dict(color=text_color, size=10),
                    margin=dict(l=40, r=20, t=35, b=35)
                )
            st.plotly_chart(fig_off2, use_container_width=True, config=config_plotly)

st.markdown('<hr style="margin: 1rem 0;">', unsafe_allow_html=True)


# ===================================================================
#                    SECCIÓN 3: DOCUMENTACIÓN DEL MODELO
# ===================================================================

st.subheader("Documentación del Modelo")

with st.expander("Modelo Conceptual"):
    pdf_url = "https://raw.githubusercontent.com/ffedezn-cloud/sintonizador-pi-tanque/main/assets/docs/modelo_conceptual.pdf"
    viewer_url = f"https://docs.google.com/viewer?url={pdf_url}&embedded=true"
    
    st.markdown(
        f'''
        <iframe src="{viewer_url}" 
                width="100%" 
                height="700px" 
                style="border: 1px solid #ddd; border-radius: 4px;">
        </iframe>
        ''',
        unsafe_allow_html=True
    )

with st.expander("Código en Octave"):
    st.markdown("""
    Código autocontenido para simular el tanque calefaccionado con control PI en Octave.
    Para utilizarlo:
    1. Copiar el código
    2. Guardarlo en un archivo con extensión .m
    3. Ejecutarlo en Octave
    """)
    
    codigo_octave = '''% Tanque calefaccionado con control de nivel
% En X están las variables de estado.
% En Y deben ir las variables que se requieren en las ODEs o que se quieren graficar.

clear all; close all; clc;

%=============== Modelo =================

% ODEs
function dX = ODEs(t,X)
  % En dX devuelve el vector columna de derivadas

  % Recupera variables X
  [L, T, Ai] = num2cell(X'){1,:};

  % Recupera variables Y
  Y = AEs(t,X);
  [A, F0, F, rho, Cp, T0, Q, Wa, e, Lsp, x, Ac] = num2cell(Y){1,:};

  % Ecuaciones diferenciales
  dL = (F0 - F)/A;
  dT = (F0*rho*Cp*(T0-T)+Q+Wa)/(A*L*rho*Cp);
  dAi = e;

  dX = [dL, dT, dAi]'; % vector columna
endfunction % ODEs

% AEs
function Y = AEs(t,X)
  % En Y devuelve el vector fila de variables requeridas por ODEs o a graficar.

  % Recupera variables X
  [L, T, Ai] = num2cell(X'){1,:};

  % Parámetros
  F0 = 2E-3; A = 0.785; Cv = 4.039E-5; rho = 1000; g = 9.81;
  Cp = 4.187E3; UAs = 4.04E3; T0 = 25; Tv = 132; Wa = 2000;
  Ab = 0.5; Kp = 2; taui = 300; Lsp = 1;

  % Ecuaciones algebraicas (escalón en Lsp)
  if t > 300
   Lsp = 0.4;
  endif

  % Controlador (acción directa: e = L - Lsp)
  e = L-Lsp;
  Ac = Ab+Kp*(e+Ai/taui);
  x = max(0,Ac); % Acota x entre 0 y 1
  x = min(x,1);
  F = Cv*x*sqrt(rho*g*L);
  Q = UAs*(Tv-T);

  Y = [A, F0, F, rho, Cp, T0, Q, Wa, e, Lsp, x, Ac];
endfunction % AEs

% Inicialización
function [tfin dt Xini LX LY] = inicializacion
  % Inicializa la simulación

  % Parámetros de simulación
  tfin = 2000; % tiempo final
  dt = 10; % paso temporal

  % Inicialización
  Lini = 1; % m
  Tini = 60; %°C
  Aiini = 0;
  Xini = [Lini; Tini; Aiini]; % Inicializa la variable de estado

  % Leyendas
  LX = {'L' 'T' 'Ai'}; % Leyendas de las variables X
  LY = {'A' 'F0' 'F' 'rho' 'Cp' 'T0' 'Q' 'Wa' 'e' 'Lsp' 'x' 'Ac'}; % Leyendas de las variables Y
endfunction % inicializar

% Análisis
function analizar(LX,LY,tpts,X,Y)
  % Análisis de resultados

  graficar({'L' 'Lsp'}, 'Nivel vs. tiempo', 's', 'm', [0 3]);
  graficar({'x' 'Ac'}, 'Apertura de válvula vs. Señal del controlador', ' ', ' ');
  graficar({'T'}, 'Temperatura vs. tiempo', 's', '°C', [0 120]);

  % Determina el valle de L
  [min_valor, indice] = min(vector('L'));
  disp('Datos del valle de nivel');
  disp(['El valor mínimo del nivel es ' num2str(min_valor) ' m.']);
  dt = tpts(2);
  disp(['Ese valor se alcanza en ' num2str(tpts(indice)) ' ± ' num2str(dt) ' s.']);
endfunction % analizar

%=============== Resolvedor (integrado) =================

function v = vector(leyenda)
  global LX LY tpts X Y
  indicex = find(strcmp(LX, leyenda));
  if length(indicex) == 1
    v = X(:,indicex);
  else
    indicey = find(strcmp(LY, leyenda));
    if length(indicey) == 1
      v = Y(:,indicey);
    else
      error('Variable no encontrada');
    endif
  endif
endfunction

function graficar(LV, titulo, rotulox, rotuloy, limitesy)
  global tpts
  colores = ['r' 'g' 'b' 'c' 'm' 'y' 'k'];
  figure; hold on;
  for i = 1:length(LV)
    plot(tpts, vector(LV{i}), colores(mod(i-1,length(LV)) + 1), 'LineWidth', 2);
  endfor
  title(titulo); xlabel(rotulox); ylabel(rotuloy);
  if nargin == 5, ylim(limitesy); endif
  grid on; legend(LV, 'Location', 'northeast');
endfunction

function [tpts X Y] = simulacion(tfin,dt,Xini)
  nts = ceil(tfin/dt + 1);
  tpts = linspace(0, tfin, nts)';
  [tpts X] = ode45(@ODEs, tpts, Xini);
  for i = 1:size(tpts,1)
    Y(i,:) = AEs(tpts(i), X(i,:)');
  endfor
endfunction

%=============== Simulación =================
clc;
disp('Resolviendo el modelo...');

global LX LY tpts X Y

[tfin dt Xini LX LY] = inicializacion;
[tpts X Y] = simulacion(tfin,dt,Xini);
analizar(LX,LY,tpts,X,Y);

disp('Simulación finalizada.');
'''
    
    st.code(codigo_octave, language="octave")
    
    st.download_button(
        label="Descargar modelo_tanque_CL.m",
        data=codigo_octave,
        file_name="modelo_tanque_CL.m",
        mime="text/plain"
    )

# Footer
st.markdown('<hr style="margin: 1rem 0;">', unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 14px; padding: 10px 0;">
        Sintonizador PI desplegado con Streamlit por Federico Franco
    </div>
    """,
    unsafe_allow_html=True
)
