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
- Pregunta: ¿Cuáles son los horarios de atención?
  Respuesta: Atendemos de lunes a viernes de 8:00 AM a 6:00 PM.

- Pregunta: ¿Dónde están ubicados?
  Respuesta: Estamos ubicados en Cl 5 #4-70, Centro, Popayán.

- Pregunta: ¿Donde puede tramitar el carnet de universitario?
  Respuesta: En DARCA, ubicado en Carrera 2 # 3N – 127, Popayán, Cauca Facultad de Educación

- Pregunta: ¿Que materias puedo ver si estudio licenciatura en matematicas, o matematicas?
  Contra pregunta: ¿En que semestre estas?
  Respuesta: primer semestre (1)- matematica generales, etc
segundo semestre (2)- Calculo I, Algebra lineal etc
tercer semestre (3)- Calculo II, etc
cuarto semestre (4)- GRupos, etc
quinto semestre (5)-Anillos, etc
sexto semestre (6)- Analisis Real, etc

"""

SYSTEM_PROMPT=f"""
Eres el asistente virtual oficial de atención al cliente.

REGLAS DE COMPORTAMIENTO:
1. Responde a las preguntas del usuario utilizando ÚNICAMENTE la información provista en la siguiente Base de Conocimiento. Verifica si la pregunta se parece y responde.
2. Si la respuesta no se encuentra en la Base de Conocimiento, responde amablemente: "Lo siento, no dispongo de esa información en este momento. Te sugiero contactar directamente a soporte humano."
3. Sé siempre amable, conciso y profesional.
4. No inventes ni supongas información que no esté explícitamente escrita abajo.

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
        <title>Chatbot Asistente Virtual</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            body { background-color: #f4f6f9; display: flex; justify-content: center; align-items: center; height: 100vh; }
            .chat-container { width: 100%; max-width: 450px; height: 600px; background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); display: flex; flex-direction: column; overflow: hidden; }
            .chat-header { background: #2b579a; color: white; padding: 15px; text-align: center; font-weight: bold; }
            .chat-box { flex: 1; padding: 15px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }
            .message { padding: 10px 14px; border-radius: 18px; max-width: 80%; font-size: 14px; line-height: 1.4; word-wrap: break-word; }
            .user { background: #0078d4; color: white; align-self: flex-end; border-bottom-right-radius: 2px; }
            .bot { background: #e9ecef; color: #333; align-self: flex-start; border-bottom-left-radius: 2px; }
            .input-area { display: flex; padding: 12px; border-top: 1px solid #ddd; background: #fff; }
            .input-area input { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 20px; outline: none; }
            .input-area button { margin-left: 8px; padding: 10px 18px; background: #0078d4; color: white; border: none; border-radius: 20px; cursor: pointer; font-weight: bold; }
            .input-area button:hover { background: #005a9e; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">Asistente Virtual</div>
            <div class="chat-box" id="chatBox">
                <div class="message bot">¡Hola! ¿En qué puedo ayudarte hoy?</div>
            </div>
            <div class="input-area">
                <input type="text" id="userInput" placeholder="Escribe tu mensaje aquí..." onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()">Enviar</button>
            </div>
        </div>

        <script>
            async function sendMessage() {
                const input = document.getElementById('userInput');
                const chatBox = document.getElementById('chatBox');
                const text = input.value.trim();

                if (!text) return;

                // Agregar mensaje usuario
                chatBox.innerHTML += `<div class="message user">${text}</div>`;
                input.value = '';
                chatBox.scrollTop = chatBox.scrollHeight;

                // Loader bot
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