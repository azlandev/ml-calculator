import Calculator from './components/Calculator';

function App() {

  return (
    <div >
      <div className='title'>
        <h1>ML-Calculator 🖩</h1>
        <hr/>
        <p>A calculator that uses machine learning to predict answers</p>
        <button onClick={()=> window.open("https://github.com/azlandev/ml-calculator", "_blank")}>GitHub</button>
      </div>

      <div className='calculator'>
        <Calculator />
      </div>
    </div>
  );
}

export default App;
