
import { GoogleGenAI, GenerateContentResponse } from "@google/genai";
import type { SentenceAnalysis } from '../types';

let ai: GoogleGenAI | null = null;

// Attempt to initialize AI client if API_KEY is present in the environment
if (process.env.API_KEY) {
  try {
    ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  } catch (e) {
    console.error("Failed to initialize Gemini Client. The API key might be malformed.", e);
    ai = null; // Ensure ai is null if initialization fails.
  }
}

/**
 * Checks if the Gemini API has been configured and initialized.
 * @returns {boolean} True if the API is ready, false otherwise.
 */
export const isApiKeyConfigured = (): boolean => {
  return !!ai;
};


const MODEL_NAME = "gemini-2.5-flash";

// Helper function for retries with exponential backoff
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

function parseGeminiResponse(responseText: string): SentenceAnalysis | null {
  let jsonStr = responseText.trim();
  // Regex to remove markdown fences (```json ... ``` or ``` ... ```)
  const fenceRegex = /^```(?:json)?\s*\n?(.*?)\n?\s*```$/si;
  const match = jsonStr.match(fenceRegex);
  if (match && match[1]) {
    jsonStr = match[1].trim();
  }

  try {
    const parsedData = JSON.parse(jsonStr);
    // Basic validation of structure format
    const isValidStructure = (elements: any[]): boolean => {
      return elements.every(el => 
        typeof el.text === 'string' &&
        typeof el.label === 'string' &&
        (!el.children || (Array.isArray(el.children) && isValidStructure(el.children)))
      );
    };

    if (parsedData && 
        typeof parsedData.fullSentence === 'string' && 
        typeof parsedData.classification === 'string' && 
        Array.isArray(parsedData.structure) &&
        isValidStructure(parsedData.structure)
    ) {
       return parsedData as SentenceAnalysis;
    }
    console.error("Parsed JSON does not match expected SentenceAnalysis structure or is invalid:", parsedData);
    return null;
  } catch (e) {
    console.error("Failed to parse JSON response:", e);
    console.error("Problematic JSON string that failed to parse:", jsonStr);
    return null;
  }
}

export const analyzeSentence = async (sentence: string): Promise<SentenceAnalysis | null> => {
  if (!ai) {
    throw new Error("La clave API de Gemini no ha sido configurada correctamente en el entorno de la aplicación.");
  }

  const prompt = `
Analiza sintácticamente la siguiente oración en español según los principios de la Nueva Gramática de la Lengua Española (NGLE) y proporciona la estructura en formato JSON. La oración es: '${sentence}'.

**OBJETIVO GENERAL:**
Producir un árbol sintáctico que refleje la estructura gramatical de la oración, identificando sintagmas, sus núcleos, funciones y las relaciones entre ellos, especialmente en oraciones compuestas y complejas.

**FORMATO JSON REQUERIDO:**
El objeto raíz debe tener:
- 'fullSentence': La oración original.
- 'classification': Clasificación detallada de la oración (ej: "Oración Simple, Enunciativa Afirmativa, Predicativa, Activa, Transitiva", "Oración Compuesta Coordinada Copulativa", "Oración Compleja Subordinada Sustantiva de CD").
- 'structure': Un array de elementos sintácticos. Generalmente, para una oración simple o la principal de una compleja/compuesta, este array contendrá dos elementos principales: 'SN Sujeto' y 'SV - Predicado verbal' (o 'SV - Predicado nominal'). Si hay un Sujeto Tácito, se incluirá un nodo 'ST'. Si hay varias oraciones coordinadas principales, cada una será un objeto 'Prop - Coordinada [Tipo]' en este array.

**ELEMENTO SINTÁCTICO ('SyntacticElement'):**
Cada elemento en 'structure' (y sus 'children') debe ser un objeto con:
- 'text': El fragmento de texto exacto de la oración que corresponde a este elemento. Para elementos como 'Sujeto Tácito', el texto puede ser "(ST)", "(Yo)", "(Él/Ella)", etc.
- 'label': La etiqueta gramatical del elemento (ver GUÍA DE ETIQUETAS más abajo).
- 'children': (Opcional) Un array de 'SyntacticElement' que son constituyentes de este elemento. Si no tiene hijos (es un elemento terminal o una palabra), omite esta propiedad o usa un array vacío.

**GUÍA DE ETIQUETAS (NGLE):**
Usa estas etiquetas de forma precisa. El primer nivel de 'structure' suele ser el Sujeto y Predicado de la oración principal.

1.  **Nivel Oracional Principal:**
    *   'SN Sujeto': Sintagma Nominal Sujeto.
    *   'SV - Predicado verbal': Sintagma Verbal Predicado Verbal.
    *   'SV - Predicado nominal': Sintagma Verbal Predicado Nominal (con verbos copulativos).
    *   'ST': Para indicar Sujeto Tácito. El 'text' puede ser "(ST)", "(Yo)", "(Nosotros)", "(Él/Ella)", etc., reflejando el sujeto omitido. Este nodo NO debe tener hijos y se coloca al mismo nivel que el predicado.

2.  **Sintagmas (Tipos y Funciones):**
    *   **Tipos de Sintagmas (como constituyentes):**
        *   'SN': Sintagma Nominal.
        *   'SAdj': Sintagma Adjetival.
        *   'SAdv': Sintagma Adverbial.
        *   'SPrep': Sintagma Preposicional.
    *   **Núcleos de Sintagmas (etiquetados con '(N)'):**
        *   'N (N)': Nombre (Núcleo de SN). Ejemplo: \`{"text": "libro", "label": "N (N)"}\`.
        *   'V (N)': Verbo (Núcleo de SV). Debe usarse esta etiqueta para el núcleo verbal. Ejemplo: \`{"text": "come", "label": "V (N)"}\`.
        *   'Adj (N)': Adjetivo (Núcleo de SAdj). Ejemplo: \`{"text": "grande", "label": "Adj (N)"}\`.
        *   'Adv (N)': Adverbio (Núcleo de SAdv). Ejemplo: \`{"text": "rápidamente", "label": "Adv (N)"}\`.
        *   'Prep (N)': Preposición (Núcleo de SPrep). Debe usarse esta etiqueta para la preposición. Ejemplo: \`{"text": "en", "label": "Prep (N)"}\`.
        *   'Pron (N)': Pronombre (Núcleo de SN, si se decide usar una etiqueta específica para pronombre núcleo, si no, 'N (N)' puede aplicar si el texto es un pronombre y actúa como tal). Para simplificar, priorizar 'N (N)' para núcleos nominales y 'Pron' para pronombres que no son núcleo de un SN mayor o tienen función propia.
    *   **Determinantes y Nexos (generalmente palabras solas):**
        *   'Det': Determinante.
        *   'nx': Nexo coordinante o subordinante que NO cumple otra función sintáctica.
        *   'PronRel': Pronombre Relativo (nexo + función dentro de la subordinada). Ejemplo: \`"text": "que", "label": "PronRel (Sujeto)"\` si 'que' es sujeto en la relativa.
        *   'AdvRel': Adverbio Relativo (nexo + función o CC). Ejemplo: \`"text": "donde", "label": "AdvRel (CCLugar)"\`.
        *   'DetRel': Determinante Relativo/Posesivo Relativo Cuyo (nexo + función Det). Ejemplo: \`"text": "cuyo", "label": "DetRel"\`.
    *   **Funciones Sintácticas (etiquetar el sintagma completo que cumple la función):**
        *   **Complemento Directo (CD):** Puede ser 'SN - CD' o 'SPrep - CD'.
            *   'SN - CD': Cuando el CD es un Sintagma Nominal.
            *   'SPrep - CD': Cuando el CD es un Sintagma Preposicional introducido por "a". En este caso, el SPrep tendrá un hijo \`{"text": "a", "label": "Prep (N)"}\` y otro hijo que será el Término (generalmente 'SN - Término').
        *   'SN - CI', 'SPrep - CI'.
        *   'SN - Atrib', 'SAdj - Atrib', 'SPrep - Atrib'.
        *   'SN - CPred', 'SAdj - CPred'.
        *   'SPrep - CRég' (Complemento de Régimen).
        *   'SPrep - CC de [Tipo]': Sintagma Preposicional C. Circunstancial (Lugar, Tiempo, Modo, Causa, Finalidad, Compañía, Instrumento, Cantidad, etc.). Ejemplo: \`"label": "SPrep - CC de Lugar"\`. El núcleo interno será \`Prep (N)\` y su término.
        *   'SAdv - CC de [Tipo]': Sintagma Adverbial C. Circunstancial. Ejemplo: \`"label": "SAdv - CC de Modo"\`. Si es una sola palabra (ej. "así"), su estructura interna será \`{"text": "así", "label": "Adv (N)"}\`.
        *   'SN - CC de [Tipo]': Sintagma Nominal C. Circunstancial (ej. "Esta mañana"). Su estructura interna contendrá \`Det\` y \`N (N)\`.
        *   'SPrep - CAg' (Complemento Agente).
        *   'SPrep - CN' (Complemento del Nombre, realizado por SPrep).
        *   'SAdj - CN' (Complemento del Nombre, cuando un SAdj modifica directamente a un N (N). No usar "Adyacente").
        *   'SPrep - CAdj' (Complemento del Adjetivo, realizado por SPrep).
        *   'SPrep - CAdv' (Complemento del Adverbio, realizado por SPrep).
        *   'SN - Voc' (Vocativo).
        *   **Término de SPrep:** El sintagma que sigue a la preposición dentro de un SPrep. El nodo de la preposición será 'Prep (N)'.
            *   'SN - Término' (con núcleo interno 'N (N)')
            *   'SAdj - Término' (con núcleo interno 'Adj (N)')
            *   'SAdv - Término' (con núcleo interno 'Adv (N)')
            *   'Oración - Subordinada Sustantiva de Término'
# ANÁLISIS DE LA ORACIÓN COMPUESTA SEGÚN LA NGLE

---

## 3. COORDINADAS
La coordinación une oraciones del mismo nivel sintáctico, sin que unas dependan de otras.  
Pueden estar ligadas por nexos coordinantes o yuxtapuestas por signos de puntuación.

| Tipo | Función | Nexos | Ejemplo |
|------|---------|-------|---------|
| **Copulativas** | Expresan suma o adición | y, e, ni, ni…ni, tanto…como, tanto…cuanto, así…como, lo mismo…que, no solo…sino que también | *Mis vecinos lo vieron **y** no le dijeron nada.* |
| **Disyuntivas** | Expresan alternativa o exclusión | o (u), o…o, bien…bien, ya…ya, sea…sea | *¿Quieres café **o** prefieres chocolate?* |
| **Adversativas** | Expresan oposición de ideas | pero, sino, sino que, aunque (≃ pero) | *Se lo dije, **pero** no me hizo caso.* |
| **Yuxtapuestas** | No hay nexo, se separan por puntuación | ( ; , : … ) | *Era tarde; nos fuimos.* |

---

## 4. SUBORDINADAS
Hay relación de dependencia entre principal y subordinada.  
La subordinada se integra como constituyente de la principal o la modifica.

### 4.1. Sustantivas
**Definición:** equivalen a un sustantivo o grupo nominal. Se pueden conmutar por “eso”, “algo”, “alguien”.  
**Notas:**  
- El nexo *que* suele ser “completivo” (no cumple función dentro de la subordinada).  
- En interrogativas/exclamativas indirectas, el nexo sí ejerce función sintáctica.

| Función | Nexos | Ejemplo |
|---------|-------|---------|
| **Sujeto** | que / interrogativas (quién, qué, cuál, cuándo, cómo, cuánto, dónde) / infinitivo | *Me gusta **que hayas estudiado** Historia.* |
| **CD** | que / interrogativas (si, quién, qué, cuál, cuándo, cómo, cuánto, dónde) / exclamativas / infinitivo | *Ellos deseaban **que les subieran el sueldo**.* |
| **Atributo** | que / infinitivo | *El problema es **que no sé la respuesta**.* |
| **Término de preposición (CI)** | prep + que | *No daba crédito **a que sacaría un diez**.* |
| **Término de preposición (C. Régimen)** | prep + que / prep + infinitivo | *No se acordaba **de que no lo había hecho**.* |
| **Término de preposición (CC)** | prep + que / prep + infinitivo | *Entraron **sin que nadie se percatara**.* |
| **Término de preposición (CN, C. Adj, C. Adv)** | prep + que / prep + infinitivo | *Estoy harto **de que tardes tanto en llegar**.* |
| **Aposición** | grupo nominal + “que” | *Os digo una cosa: **que tengáis cuidado*** |

---

### 4.2. Relativas
**Definición:** modifican a un sustantivo (antecedente), como lo haría un adjetivo.  
**Notas:**  
- El nexo cumple función dentro de la subordinada (*que me besó* → “que” = sujeto).  
- Pueden ser **especificativas** (sin comas, restringen el referente) o **explicativas** (con comas, aportan información adicional).  
- También existen las relativas **libres** (sin antecedente expreso), que funcionan como sujeto, CD, CCL, etc.

| Tipo | Nexos | Ejemplo | Función |
|------|-------|---------|---------|
| **Con antecedente expreso** | que, quien, el/la cual, los/las cuales, donde, como, cuando, cuyo/a/os/as | *El chico **que me besó*** | CN del antecedente |
| **Sin antecedente expreso (libres)** | quien, cuanto, donde, como, cuando | *Quien dice eso, miente.* | Sujeto / CD / CCL / CCM |
| **Sin antecedente expreso (semilibres)** | el que, la que, los que… | *El que la hace, la paga.* | Sujeto |

---

### 4.3. Construcciones (adverbiales tradicionales)
**Definición:** la NGLE llama “construcciones” a las subordinadas que expresan valores circunstanciales de tiempo, modo, causa, finalidad, etc.  
**Notas:**  
- Funcionan globalmente como complementos circunstanciales o modificadores.  
- Se reconocen porque responden a preguntas del tipo “¿cuándo?”, “¿cómo?”, “¿por qué?”, “¿para qué?”, etc.

| Tipo | Función | Nexos | Ejemplo |
|------|---------|-------|---------|
| **Temporales** | CCT | mientras, hasta que, desde que, antes de que, luego de que, hacer que, al + infinitivo, gerundio, participio | *Mientras que estudias, voy a la tienda.* |
| **Modales** | CCM | según, conforme, tal y como, gerundio | *Hazlo según indica el manual.* |
| **Causales** | Causa | porque, ya que, a causa de que, dado que, puesto que, gracias a que, etc. | *Miguel se fue porque estaba cansado.* |
| **Finales** | Finalidad | para que, a fin de que, con objeto de que, para + infinitivo | *Expliqué despacio **para que** lo entendieran.* |
| **Ilativas** | Consecuencia lógica | así que, de modo que, de manera que, luego, conque, pues | *Está lloviendo, **así que** coge un paraguas.* |
| **Consecutivas** | Consecuencia intensiva | tan…que, tanto…que, tal…que | *Ha corrido tanto que ha llegado exhausto.* |
| **Concesivas** | Oposición | aunque, pese a que, a pesar de que, si bien, aun + gerundio/participio | *Aunque sea difícil, lo conseguiremos.* |
| **Condicionales** | Condición | si, siempre que, a condición de que, en el caso de que, gerundio, participio | *Si apruebo el curso, me regalan un móvil.* |
| **Comparativas** | Comparación | tan…como, tanto…como, más…que, menos…que, igual…que | *Ángela ha leído más novelas que José.* |
| **Superlativas** | Grado máximo | el/la/los/as más…que, menos…que, tan…como | *Este es el chiste más gracioso que he oído.* |



5.  **Otras Etiquetas:**
    *   'Interj': Interjección.
    *   'Perífrasis Verbal': Etiquetar el conjunto como 'V (N)' y en 'text' poner la perífrasis completa. Ej: \`{"text": "va a llover", "label": "V (N)"}\`.

**CONSIDERACIONES IMPORTANTES:**
- **Jerarquía:** Los hijos de un nodo deben ser sus constituyentes directos.
- **Núcleos (N):** Asegúrate de que los núcleos de los sintagmas (SN, SAdj, SAdv, SPrep, SV) se identifiquen consistentemente con la etiqueta \`(N)\` apropiada: \`N (N)\`, \`Adj (N)\`, \`Adv (N)\`, \`Prep (N)\`, \`V (N)\`.
- **Texto Completo:** La concatenación de los 'text' de los nodos terminales debe reconstruir la oración.

**Corrección al EJEMPLO DE SALIDA (para "El perro que ladra no muerde"):**
El N (N) no debe tener como hijo a la Oración Relativa. La Oración Relativa es un modificador del Nombre, por tanto, es hermana del N (N) dentro del SN.
\`\`\`json
{
  "fullSentence": "El perro que ladra no muerde",
  "classification": "Oración Compleja, Enunciativa Negativa, Predicativa, Activa, Intransitiva, con Subordinada Relativa Especificativa",
  "structure": [
    {
      "text": "El perro que ladra",
      "label": "SN Sujeto",
      "children": [
        { "text": "El", "label": "Det" },
        { "text": "perro", "label": "N (N)" },
        {
          "text": "que ladra",
          "label": "Oración - Subordinada Relativa Especificativa", 
          "children": [
            { "text": "que", "label": "PronRel (Sujeto)" },
            {
              "text": "ladra",
              "label": "SV - Predicado verbal",
              "children": [
                { "text": "ladra", "label": "V (N)" }
              ]
            }
          ]
        }
      ]
    },
    {
      "text": "no muerde",
      "label": "SV - Predicado verbal",
      "children": [
        {
          "text": "no",
          "label": "SAdv - CC de Negación",
          "children": [
            { "text": "no", "label": "Adv (N)" }
          ]
        },
        { "text": "muerde", "label": "V (N)" }
      ]
    }
  ]
}
\`\`\`

Proporciona SOLO el objeto JSON como respuesta. No incluyas explicaciones adicionales fuera del JSON.
`;

  console.log("Gemini API Prompt length:", prompt.length);

  const MAX_RETRIES = 3;
  let attempt = 0;
  let lastError: any = null;

  while (attempt < MAX_RETRIES) {
    try {
      const response: GenerateContentResponse = await ai.models.generateContent({
        model: MODEL_NAME,
        contents: prompt,
        config: {
          responseMimeType: "application/json",
          // temperature: 0.3 (opcional, para mayor determinismo si es necesario)
        },
      });

      const responseText = response.text;
      if (!responseText) {
          console.error("Gemini response text is empty on attempt " + (attempt + 1));
          return null; 
      }
      return parseGeminiResponse(responseText);

    } catch (error: any) {
      lastError = error;
      const errorMessage = String(error.message || error).toLowerCase();
      console.error(`Error calling Gemini API (Attempt ${attempt + 1}/${MAX_RETRIES}):`, error);

      // Conditions for retry
      const isRetryableError = 
           errorMessage.includes("rpc failed") || 
           errorMessage.includes("xhr error") ||
           errorMessage.includes("fetch failed") || 
           errorMessage.includes("network error") || // Generic network error
           errorMessage.includes("timeout") || // Explicit timeout
           (error.status && error.status >= 500); // HTTP 5xx errors if available

      if (isRetryableError && attempt < MAX_RETRIES - 1) {
        const delayTime = 1000 * Math.pow(2, attempt); // Exponential backoff: 1s, 2s
        console.log(`Retrying in ${delayTime / 1000}s...`);
        await delay(delayTime);
        attempt++;
      } else {
        // Non-retryable error or max retries reached
        if (errorMessage.includes("api_key_invalid") || (error.message && error.message.includes("[400]"))) {
            throw new Error("La clave API de Gemini no es válida o ha expirado. Por favor, verifica la configuración.");
        }
        if (errorMessage.includes("fetch failed") || errorMessage.includes("inet") || errorMessage.includes("network")) {
             throw new Error("Error de red al contactar el servicio de Gemini. Verifica tu conexión a internet e inténtalo de nuevo.");
        }
        // Throw the last encountered error
        throw new Error(`Error al procesar la solicitud con Gemini después de ${MAX_RETRIES} intentos: ${lastError.message || lastError}`);
      }
    }
  }
  return null; // Should not be reached if error thrown, but as a fallback
};
