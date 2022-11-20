import '../styles.css';
import DigitButton from './DigitButton';
import OperationButton from './OperationButton';
import React, { useReducer, useState } from 'react';

export const ACTIONS = {
    ADD_DIGIT: 'add-digit',
    DELETE_DIGIT: 'delete-digit',
    CLEAR: 'clear',
    CHOOSE_OPERATION: 'choose-operation',
    EVALUATE: 'evaluate'
};

const initialState = {
    curNum: null,
    prevNum: null,
    operation: null,
    overwrite: false
};

function reducer(state, { type, payload }) {
    switch(type) {
        case ACTIONS.ADD_DIGIT:
            if(state.overwrite) {
                return {
                    ...state,
                    curNum: payload.digit,
                    prevNum: null,
                    overwrite: false
                };
            }
            if(payload.digit === '0' && state.curNum === '0') return state;
            if(payload.digit === '.' && state.curNum.includes('.')) return state;
  
            return {
                ...state,
                curNum: `${state.curNum || ''}${payload.digit}`
            };
        case ACTIONS.CHOOSE_OPERATION:
            if(state.curNum == null && state.prevNum == null) return state;
            if(state.curNum == null) {
                return {
                    ...state,
                    operation: payload.operation
                };
            }
            if(state.prevNum == null) {
                return {
                    ...state,
                    operation: payload.operation,
                    prevNum: state.curNum,
                    curNum: null
                };
            }
            return state;
        case ACTIONS.CLEAR:
            return {};
        case ACTIONS.DELETE_DIGIT:
            if(state.overwrite) {
                return {
                    ...state,
                    overwrite: false,
                    curNum: null
                };
            }
            if(state.curNum == null) return state;
            if(state.curNum.length === 1) {
                return {
                    ...state,
                    curNum: null
                };
            }
            return {
                ...state,
                curNum: state.curNum.slice(0, -1)
            };
        case ACTIONS.EVALUATE:
            if(state.curNum == null || state.prevNum == null || state.operation == null) return state;
                return {
                    ...state,
                    overwrite: true,
                    prevNum: 'Accuracy: ' + payload.accuracy.toString().match(/^-?\d+(?:\.\d{0,2})?/)[0] + '%',
                    operation: null,
                    curNum: payload.prediction
                };
        default:
            console.log('Invalid ACTION');
            return;
    }
}

function evaluate(curState, dispatch, setLoading) {
    setLoading(true);
    
    const x1 = parseFloat(curState.prevNum);
    const x2 = parseFloat(curState.curNum);
    if(isNaN(x1) || isNaN(x2)) return '';
  
    let actualAnswer = 0;
    switch(curState.operation){
        case '+':
            actualAnswer = x1 + x2;
            break;
        case '-':
            actualAnswer = x1 - x2;
            break;
        case '×':
            actualAnswer = x1 * x2;
            break;
        case '÷':
            actualAnswer = x1 / x2;
            break;
        default:
            console.log("Invalid operator");
            return;
    }
  
    const requestOptions = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            x1: x1,
            x2: x2,
            operator: curState.operation
        })
    };
  
    fetch('/calculate', requestOptions).then(res => res.json()).then(
        data => {dispatch({ 
            type: ACTIONS.EVALUATE, 
            payload: {
                prediction: data.prediction,
                accuracy: 100 - (Math.abs(data.prediction - actualAnswer)/Math.max(Math.abs(data.prediction), Math.abs(actualAnswer)))
            }
        })
        setLoading(false);
        }
    )
}

export default function Calculator() {
    const [state, dispatch] = useReducer(reducer, initialState);
    const [loading, setLoading] = useState(false); 
  
    return (
        <div className='calculator-grid'>
            <div className='output'>
                <div className='prev-num'>{state.prevNum} {state.operation}</div>
                <div className='cur-num'>{loading ? <div className='loading-spinner'></div> : state.curNum}</div>
            </div>
            <button className='span-two' 
            onClick={() => dispatch({ type: ACTIONS.CLEAR })}>AC</button>
            <button
            onClick={() => dispatch({ type: ACTIONS.DELETE_DIGIT })}>Del</button>
            <OperationButton operation='÷' dispatch={dispatch} />
            <DigitButton digit='1' dispatch={dispatch} />
            <DigitButton digit='2' dispatch={dispatch} />
            <DigitButton digit='3' dispatch={dispatch} />
            <OperationButton operation='×' dispatch={dispatch} />
            <DigitButton digit='4' dispatch={dispatch} />
            <DigitButton digit='5' dispatch={dispatch} />
            <DigitButton digit='6' dispatch={dispatch} />
            <OperationButton operation='+' dispatch={dispatch} />
            <DigitButton digit='7' dispatch={dispatch} />
            <DigitButton digit='8' dispatch={dispatch} />
            <DigitButton digit='9' dispatch={dispatch} />
            <OperationButton operation='-' dispatch={dispatch} />
            <DigitButton digit='.' dispatch={dispatch} />
            <DigitButton digit='0' dispatch={dispatch} />
            <button className='span-two' 
            onClick={async () => evaluate(state, dispatch, setLoading)}>=</button>
        </div>
    );
}
