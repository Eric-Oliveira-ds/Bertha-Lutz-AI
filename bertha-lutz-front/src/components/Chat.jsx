import { useState } from "react";
import axios from "axios";

function Chat(){

  const [userId,setUserId] = useState("");
  const [message,setMessage] = useState("");
  const [response,setResponse] = useState("");

  const sendMessage = async (e)=>{

    e.preventDefault();

    try{

      const res = await axios.post(
        "http://localhost:8000/chat",
        {
          user_id:userId,
          message:message
        }
      );

      setResponse(res.data.response);

    }catch(error){

      console.log(error);

    }

  };

  return(

    <div>

      <h2>Chat</h2>

      <form onSubmit={sendMessage}>

        <div>
          User ID
          <input
            type="number"
            value={userId}
            onChange={(e)=>setUserId(e.target.value)}
          />
        </div>

        <div>
          Mensagem
          <textarea
            rows="4"
            value={message}
            onChange={(e)=>setMessage(e.target.value)}
          />
        </div>

        <button type="submit">
          Enviar
        </button>

      </form>

      {response && (
        <div>
          <h3>Resposta do agente</h3>
          <p>{response}</p>
        </div>
      )}

    </div>

  )

}

export default Chat;
