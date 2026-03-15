import Register from "./components/Register";
import Chat from "./components/Chat";

function App() {
  return (
    <div style={{ padding: "40px", fontFamily: "Arial" }}>
      <h1>Bertha Lutz AI</h1>

      <p>Sistema de acompanhamento da saúde da mulher</p>

      <hr />

      <Register />

      <hr />

      <Chat />

    </div>
  );
}

export default App;
