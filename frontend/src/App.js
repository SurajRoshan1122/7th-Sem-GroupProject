import React from "react";
import { BrowserRouter as Router,Routes, Route} from "react-router-dom";
import Home from "./Home";
import Next from "./Next";
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/next" element={<Next/>} />
      </Routes>
    </Router>
  );
}

export default App;
