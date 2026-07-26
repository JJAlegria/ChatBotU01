import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# 1. Cargar variables de entorno
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 2. Inicializar FastAPI
app = FastAPI(title="Chatbot Web API")

# Habilitar CORS para que tu interfaz o cualquier sitio web externo pueda hacerle peticiones
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Base de Conocimiento y Prompt de Sistema
BASE_DE_CONOCIMIENTO = """
- Pregunta: ¿Cuáles son los horarios de atención? <hora, abrir>
  Respuesta: Atendemos de lunes a viernes de 8:00 AM a 6:00 PM.

- Pregunta: ¿Dónde están ubicados? <lugar, estan, encontrar, ubicar, localizar>
  Respuesta: Estamos ubicados en Cl 5 #4-70, Centro, Popayán.

- Pregunta: ¿Donde puede tramitar el carnet de universitario? <sacar, obtener, verificar>
  Respuesta: En DARCA, ubicado en Carrera 2 # 3N – 127, Popayán, Cauca Facultad de Educación

- Pregunta: ¿Que materias puedo ver el siguiente semestre? <matricular, estudiar>
  Contra pregunta: ¿Que carrera estudias?
  Contra pregunta: ¿Que semestre estas cursando?
  Respuesta: -<matematicas> En matematicas puedes matricular
                 <primer, 1> matematica generales, etc
                 <segundo, 2> Calculo I, Algebra lineal etc
                 <tercer, 3>  Calculo II, programcion basica.
             -<Licenciatura matematica> En licenciatura en matematicas puedes matricular
                 <primer, 1> matematica generales, etc
                 <segundo, 2> Calculo I, Algebra lineal, pensamiento matemático etc
                 <tercer, 3>  Calculo II, programcion basica.
            -<Ingeniria fisica, Ing, fisica> En ingenieria fisica puedes matricular
                 <primer, 1> Calculo I, etc
                 <segundo, 2> Calculo II, Algebra lineal, pensamiento matemático etc
                 <tercer, 3>  Calculo III, Electromagnetismo.
-Pregunta: ¿Quien es el decano? <indicar, dime,dame, nombre>
Contra pregunta: ¿en que facultad estudias?
Respuesta: <educacion, ciencias> El decano es el Dr Jairo Roa
Respuesta: <electronica, sistemas, telecomunicaciones> El decano es Ph.D Francisco José Pino Correa.
-Pregunta: ¿Quien es el jefe de departamento? <indicar, dime, dame, nombre>
Contra pregunta: ¿Cual departamento?
Respuesta: <matematicas, licenciatura> El jefe de departamento de matemáticas es el Dr Ramiro Miguel Acevedo
Respuesta: <electronica, sistemas, telecomunicaciones> El jefe de departamento de electronica y telecomunicacione es el Ingeniero Alvaro Rene Restrepo 

"""

SYSTEM_PROMPT=f"""
Eres el asistente virtual oficial de atención al cliente.

REGLAS DE COMPORTAMIENTO:
1. Responde a las preguntas del usuario utilizando UNICAMENTE la información provista en la siguiente Base de Conocimiento, usa esto para COSTRUIR tus las respuestas. Verifica si la pregunta se parece y responde.
2. Si la respuesta no se encuentra en la Base de Conocimiento, responde amablemente: "Lo siento, no dispongo de esa información en este momento. Te sugiero contactar directamente a soporte humano."
3. Sé siempre amable, conciso y profesional.
4. No inventes ni supongas información que no esté explicitamente en la en Base de Conocimiento.
FORMATO BASE DE CONOCIMIENTOS:
5. La informacion tiene el siguiente formato, la info entre <---,---> son las palabras clave, usalas para retroalimentar la informacion que suministra el usuario y las respuesta que das
    -<palabras clave> pregunta: aqui va una pregunta generica suministrada por el usuario.
    -<palabras clave> contra pregunta: esta pregunta la hace el chat para completar informacion
    -<palabras clave> respuesta: aqui va la respuesta, se comporta como un arbol mediante la identacion
                    <palabra clave> respuesta por palabra clave
6. EL formato respeta las identaciones y navega por las respuestas como un árbol usando las palabras clve y las identaciones.
FORMATO RESPUESTA:
6. Las <palabras clave> estan en las preguntas para que puedas guiarte entre las respuestas. Buscalas en las preguntas y usalas  
7. El formato de tus respuestas es SOLO texto. NO uses los caracteres "<, >" en tusrepsuestas. EJEMPPLO: En el segundo semestre de ingenieria fisica puedes matricular: Calculo II, Algebra lineal

BASE DE CONOCIMIENTO:
{BASE_DE_CONOCIMIENTO}
"""


# 4. Modelos de datos Pydantic
class MessageRequest(BaseModel):
    message: str

class MessageResponse(BaseModel):
    reply: str

# 5. Endpoint de la API (/chat)
@app.post("/chat", response_model=MessageResponse)
async def chat_endpoint(payload: MessageRequest):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Servidor no configurado con GROQ_API_KEY.")

    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.2,
            groq_api_key=api_key
        )
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=payload.message)
        ]
        
        response = await llm.ainvoke(messages)
        return MessageResponse(reply=response.content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la IA: {str(e)}")

# 6. Interfaz Web Integrada (Página HTML al visitar la raíz "/")
@app.get("/", response_class=HTMLResponse)
async def serve_web_ui():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Asistente Virtual - Unicauca</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            body { background-color: #f0f2f5; display: flex; justify-content: center; align-items: center; height: 100vh; }
            
            /* Contenedor más compacto (380px x 520px) */
            .chat-container { 
                width: 100%; 
                max-width: 380px; 
                height: 520px; 
                background: white; 
                border-radius: 12px; 
                box-shadow: 0 4px 20px rgba(0,0,0,0.15); 
                display: flex; 
                flex-direction: column; 
                overflow: hidden; 
                border: 1px solid #e0e0e0;
            }
            
            /* Encabezado con colores institucionales (Azul y Rojo Unicauca) */
            .chat-header { 
                background: linear-gradient(135deg, #0D2C54 0%, #8A151B 100%); 
                color: white; 
                padding: 12px 15px; 
                display: flex; 
                align-items: center; 
                gap: 12px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }

            /* Espacio y contenedor para el Logo */
            .chat-header img {
                height: 38px;
                width: auto;
                background: white;
                padding: 3px 6px;
                border-radius: 4px;
                object-fit: contain;
            }

            .header-title {
                display: flex;
                flex-direction: column;
                text-align: left;
            }

            .header-title h1 {
                font-size: 15px;
                font-weight: 600;
                line-height: 1.2;
            }

            .header-title span {
                font-size: 11px;
                opacity: 0.85;
            }

            .chat-box { flex: 1; padding: 12px; overflow-y: auto; display: flex; flex-direction: column; gap: 8px; background-color: #fafafa; }
            .message { padding: 9px 13px; border-radius: 16px; max-width: 82%; font-size: 13px; line-height: 1.4; word-wrap: break-word; }
            
            /* Burbuja Usuario: Rojo Unicauca */
            .user { background: #8A151B; color: white; align-self: flex-end; border-bottom-right-radius: 3px; }
            
            /* Burbuja Bot: Azul tenue */
            .bot { background: #eef2f7; color: #1a2530; align-self: flex-start; border-bottom-left-radius: 3px; border: 1px solid #e1e8f0; }
            
            .input-area { display: flex; padding: 10px; border-top: 1px solid #e6e6e6; background: #fff; gap: 6px; }
            .input-area input { flex: 1; padding: 8px 14px; border: 1px solid #ccc; border-radius: 18px; outline: none; font-size: 13px; }
            .input-area input:focus { border-color: #0D2C54; }
            
            /* Botón Enviar: Azul Unicauca con hover a Rojo */
            .input-area button { 
                padding: 8px 15px; 
                background: #0D2C54; 
                color: white; 
                border: none; 
                border-radius: 18px; 
                cursor: pointer; 
                font-weight: 600; 
                font-size: 13px;
                transition: background 0.2s;
            }
            .input-area button:hover { background: #8A151B; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <div class="header-title">
                    <h1>Asistente Unicauca</h1>
                    <span>Atención al Usuario</span>
                </div>
            </div>
            
            <div class="chat-box" id="chatBox">
                <div class="message bot">¡Hola! Bienvenido al asistente virtual de la Universidad del Cauca. ¿En qué puedo ayudarte hoy?</div>
            </div>
            
            <div class="input-area">
                <input type="text" id="userInput" placeholder="Escribe tu mensaje..." onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()">Enviar</button>
            </div>
        </div>

        <script>
            async function sendMessage() {
                const input = document.getElementById('userInput');
                const chatBox = document.getElementById('chatBox');
                const text = input.value.trim();

                if (!text) return;

                // Agregar mensaje del usuario
                chatBox.innerHTML += `<div class="message user">${text}</div>`;
                input.value = '';
                chatBox.scrollTop = chatBox.scrollHeight;

                // Indicador de "Escribiendo..."
                const loadingDiv = document.createElement('div');
                loadingDiv.className = 'message bot';
                loadingDiv.innerText = 'Escribiendo...';
                chatBox.appendChild(loadingDiv);
                chatBox.scrollTop = chatBox.scrollHeight;

                try {
                    const res = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: text })
                    });
                    const data = await res.json();
                    loadingDiv.innerText = data.reply || data.detail || 'Error al obtener respuesta.';
                } catch (err) {
                    loadingDiv.innerText = 'Error de conexión con el servidor.';
                }
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            function handleKeyPress(e) {
                if (e.key === 'Enter') sendMessage();
            }
        </script>
    </body>
    </html>
    """
