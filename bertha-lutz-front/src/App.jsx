import Register from "./components/Register";
import Landing from "./components/Landing";

function App() {
  return (
    <div style={{ padding: "40px", fontFamily: "Arial" }}>
      <h1 style={{ color: "#2c7be5" }}>Bertha Lutz AI</h1>

      <h2 style={{ color: "#ff4fa3" }}>Acompanhamento de saúde da mulher</h2>
      <Landing />
      {/* <Register /> */}
    </div>
  );
}

export default App;
