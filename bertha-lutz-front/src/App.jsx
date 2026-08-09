import Landing from "./components/Landing";

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>
          Bertha Lutz <span>AI</span>
        </h1>
        <p>Acompanhamento de saúde da mulher</p>
      </header>
      <Landing />
    </div>
  );
}

export default App;