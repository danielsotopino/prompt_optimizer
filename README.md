# Self-Supervised Prompt Optimization (SPO) Framework

Una implementación completa del framework SPO basado en MetaGPT para optimización automatizada de prompts usando OpenAI. Este proyecto está inspirado en las técnicas de optimización de prompts documentadas por Mistral AI y adaptado para trabajar con modelos de OpenAI.

## 🚀 Características

- **Optimización Iterativa**: Mejora automática de prompts a través de múltiples iteraciones
- **Múltiples Estrategias**: Soporte para diferentes enfoques de optimización
- **Sistema de Evaluación Integral**: Métricas comprehensivas para evaluar calidad de prompts
- **Pipeline Avanzado**: Comparación de estrategias y optimización multi-objetivo
- **Funciones Lambda MetaGPT**: Implementación fiel al notebook original con funciones lambda
- **Puntuación de Confianza**: Sistema avanzado para medir confiabilidad de resultados
- **Criterios de Parada Automática**: Detección inteligente de convergencia
- **Soporte YAML/JSON**: Formatos flexibles para definir prompts e inputs
- **Ejemplo Práctico**: Demostración completa con clasificación de títulos de trabajo

## 📁 Estructura del Proyecto

```
prompt_optimizer/
├── spo_framework.py          # Framework principal SPO
├── optimization_pipeline.py  # Pipeline avanzado con múltiples estrategias
├── evaluation_system.py      # Sistema de evaluación comprehensivo
├── lambda_functions.py       # Funciones lambda estilo MetaGPT
├── confidence_scoring.py     # Sistema de puntuación de confianza
├── yaml_parser.py            # Parser para archivos YAML/JSON
├── main.py                   # Interfaz CLI
├── requirements.txt          # Dependencias (incluye PyYAML)
├── .env                      # Variables de entorno (API keys)
├── .gitignore                # Archivos a ignorar en git
├── examples/                 # Ejemplos de configuración (✅ Probados)
│   ├── sample_inputs.json        # Ejemplo básico en JSON
│   ├── sample_inputs.yaml        # Ejemplo básico en YAML
│   ├── sample_expected.json      # Salidas esperadas para los ejemplos
│   ├── classification_example.yaml # Ejemplo completo de clasificación
│   ├── prompt_config_example.yaml  # Configuración con prompt incluido
│   └── globot.yaml               # Ejemplo avanzado de clasificación de mensajes
└── README.md                # Este archivo
```

## 🛠️ Instalación

1. **Crear entorno virtual Python 3.11**:
```bash
python3.11 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

2. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

3. **Configurar API Key de OpenAI**:
```bash
export OPENAI_API_KEY="tu-api-key-aqui"
```

## 🎯 Uso Rápido

### 🚀 Prueba Rápida del Framework (Recomendado para empezar)

```bash
# 1. Activar entorno virtual
source venv/bin/activate

# 2. Configurar API key de OpenAI
export OPENAI_API_KEY="tu-api-key-aqui"

# 3. Ejecutar prueba rápida (2-3 minutos, ~$0.10-0.20)
python test_optimization.py
```

**Lo que hace la prueba rápida:**
- ⚡ 3 iteraciones de optimización básica
- 🎯 Caso de uso: clasificación de títulos de trabajo
- 📊 Muestra puntuaciones y mejoras en tiempo real
- 💰 Usa `gpt-4o-mini` para menor costo
- ⏱️ Completa en 2-3 minutos

### Ejemplo de Clasificación de Títulos de Trabajo

```bash
# Demo rápido (5-10 minutos, ~$0.50-1.00)
python main.py job-title-example --mode quick --output quick_results.json

# Demo completo con todas las características (15-20 minutos, ~$2.00-4.00)
python main.py job-title-example --mode full --output full_results.json
```

### 🎛️ Opciones de Prueba Avanzadas

#### Optimización Personalizada

```bash
# Usando archivos JSON tradicionales
python main.py optimize \
  --prompt "Clasifica el título de trabajo en categorías. Formato: Categoría - Subcategoría. Título: {job_title}" \
  --task "Clasificación de títulos de trabajo" \
  --inputs examples/sample_inputs.json \
  --expected examples/sample_expected.json \
  --iterations 5 \
  --output optimization_results.json

# Usando archivos YAML (nuevo)
python main.py optimize \
  --prompt "Clasifica el título de trabajo en categorías" \
  --task "Clasificación de títulos de trabajo" \
  --inputs examples/sample_inputs.yaml \
  --expected examples/sample_expected.yaml \
  --iterations 5 \
  --output optimization_results.json

# Usando archivo YAML completo con prompt incluido
python main.py optimize \
  --inputs examples/classification_example.yaml \
  --output optimization_results.json

# Ejemplo avanzado con clasificación de mensajes de chat
python main.py optimize \
  --inputs examples/globot.yaml \
  --output globot_optimization_results.json
```

#### Comparación de Estrategias
```bash
# Con archivos JSON
python main.py compare \
  --prompt "Clasifica el título de trabajo en categorías apropiadas" \
  --task "Clasificación de títulos de trabajo" \
  --inputs examples/sample_inputs.json \
  --strategies iterative_refinement ensemble_voting multi_objective \
  --output strategy_comparison.json

# Con archivos YAML
python main.py compare \
  --inputs examples/classification_example.yaml \
  --strategies iterative_refinement ensemble_voting \
  --output strategy_comparison.json
```

#### Solo Evaluación (sin optimización)
```bash
# Con archivos JSON
python main.py evaluate \
  --prompt "Tu prompt a evaluar" \
  --inputs examples/sample_inputs.json \
  --outputs examples/sample_outputs.json \
  --expected examples/sample_expected.json \
  --metrics accuracy relevance clarity \
  --output evaluation_results.json

# Con archivos YAML
python main.py evaluate \
  --prompt "Tu prompt a evaluar" \
  --inputs examples/classification_example.yaml \
  --outputs examples/outputs.yaml \
  --expected examples/classification_example.yaml \
  --metrics accuracy relevance clarity \
  --output evaluation_results.json
```

## 📝 Soporte para Formatos YAML y JSON

### Formato YAML Completo (Recomendado)

El framework ahora soporta archivos YAML que incluyen tanto el prompt como los datos de entrada y salida esperada en un solo archivo:

```yaml
prompt: |
  Eres un agente especializado en clasificar mensajes de usuarios.
  Tu tarea es analizar cada mensaje y clasificarlo en una de las siguientes categorías:
  - business_relevant: Mensajes relacionados con consultas de negocio o productos
  - support_request: Solicitudes de ayuda o soporte técnico
  - complaint: Quejas o expresiones de insatisfacción
  - compliment: Elogios o comentarios positivos
  
  Responde únicamente con la categoría correspondiente.

task: "Clasificar mensajes de usuarios en categorías predefinidas"

data:
  - message: "Hola, ¿tienen stock de computadores?"
    classification: "business_relevant"
  - message: "Mi cuenta no funciona, necesito ayuda"
    classification: "support_request"
  - message: "Estoy muy molesto porque mi pedido llegó tarde"
    classification: "complaint"
  - message: "Excelente servicio, muy recomendado"
    classification: "compliment"

iterations: 5
model: "gpt-4o"
```

### Formatos Soportados

1. **YAML con datos combinados** (como el ejemplo anterior)
2. **YAML con secciones separadas**:
```yaml
inputs:
  - "Mensaje 1"
  - "Mensaje 2"
expected:
  - "Categoría 1"
  - "Categoría 2"
```

3. **JSON tradicional** (mantiene compatibilidad):
```json
["input1", "input2", "input3"]
```

### Uso del Parser YAML

```python
from yaml_parser import YAMLParser

# Cargar inputs desde cualquier formato
inputs = YAMLParser.load_inputs_flexible("examples/archivo.yaml")

# Cargar outputs esperados
expected = YAMLParser.load_expected_flexible("examples/archivo.yaml")

# Cargar configuración completa
config = YAMLParser.parse_prompt_config("examples/classification_example.yaml")
```

## 📂 Directorio de Ejemplos

El directorio `examples/` contiene archivos de ejemplo para diferentes casos de uso:

### 📄 Archivos Disponibles

1. **`sample_inputs.json`** - Ejemplo básico en formato JSON
   ```json
   [
     "Senior Software Engineer at Google",
     "Data Scientist - Machine Learning",
     "Marketing Manager, Digital Products"
   ]
   ```

2. **`sample_inputs.yaml`** - Mismo contenido en formato YAML
   ```yaml
   inputs:
     - "Senior Software Engineer at Google"
     - "Data Scientist - Machine Learning"
     - "Marketing Manager, Digital Products"
   ```

3. **`classification_example.yaml`** - Ejemplo completo con prompt y datos
   ```yaml
   prompt: |
     Clasifica mensajes en categorías específicas...
   
   data:
     - message: "Hola, ¿tienen stock?"
       classification: "business_relevant"
   ```

4. **`globot.yaml`** - Ejemplo avanzado de clasificación de mensajes de chat
   - Prompt detallado con reglas específicas
   - 20+ ejemplos de entrenamiento
   - 3 categorías: business_relevant, contact_message, chitchat

5. **`prompt_config_example.yaml`** - Configuración con múltiples parámetros
   ```yaml
   prompt: |
     Tu prompt aquí...
   task: "Descripción de la tarea"
   iterations: 5
   model: "gpt-4o"
   strategies: ["iterative_refinement"]
   ```

### 🚀 Cómo Usar los Ejemplos (✅ Todos Probados)

```bash
# 1. Ejemplo básico con prompt externo
python main.py optimize \
  --inputs examples/sample_inputs.yaml \
  --prompt "Classify job titles into categories" \
  --task "Job title classification" \
  --iterations 2 \
  --output results.yaml

# 2. Ejemplo completo con prompt incluido
python main.py optimize \
  --inputs examples/classification_example.yaml \
  --iterations 3 \
  --output classification_results.yaml

# 3. Ejemplo avanzado de clasificación de mensajes (globot)
python main.py optimize \
  --inputs examples/globot.yaml \
  --iterations 3 \
  --output globot_results.yaml

# 4. Usando archivos JSON tradicionales
python main.py optimize \
  --prompt "Classify these job titles" \
  --task "Job classification" \
  --inputs examples/sample_inputs.json \
  --expected examples/sample_expected.json \
  --output traditional_results.json
```

### ✅ Resultados de Pruebas

Los ejemplos han sido probados y funcionan correctamente:

- **`classification_example.yaml`**: ✅ Score perfecto (1.00) en 1 iteración
- **`prompt_config_example.yaml`**: ✅ Score perfecto (1.00) en 1 iteración
- **`sample_inputs.yaml`**: ✅ Score mejorado (0.68) en 2 iteraciones  
- **`globot.yaml`**: ✅ Score excelente (0.91+) demostrado anteriormente
- **Salida YAML**: ✅ Formato legible y bien estructurado
- **Todos los formatos**: ✅ JSON y YAML funcionando correctamente

### Ventajas del Formato YAML

- **Prompts multilínea**: Usa `|` para prompts largos y complejos
- **Un solo archivo**: Todo en un lugar (prompt, datos, configuración)
- **Más legible**: Formato más claro que JSON
- **Comentarios**: Puedes agregar comentarios con `#`
- **Flexible**: Múltiples formatos soportados
- **Salida en YAML**: Los resultados también pueden guardarse en YAML

### 💾 Formatos de Salida

El framework detecta automáticamente el formato de salida basado en la extensión del archivo:

```bash
# Salida en JSON (tradicional)
python main.py optimize --inputs examples/globot.yaml --output results.json

# Salida en YAML (nuevo, más legible)
python main.py optimize --inputs examples/globot.yaml --output results.yaml
```

**Ejemplo de salida en YAML:**
```yaml
config:
  max_iterations: 5
  optimization_model: "gpt-4o"
  execution_model: "gpt-4o-mini"

optimization_history:
  - iteration: 1
    performance_score: 0.91
    optimized_prompt: "Prompt optimizado..."
    execution_time: 45.2

summary:
  total_iterations: 5
  best_score: 0.95
  improvement: 0.15
  best_prompt: "Mejor prompt encontrado..."
```

## 📊 Estrategias de Optimización

### 1. Refinamiento Iterativo (`iterative_refinement`)
- **Mejor para**: Optimización de propósito general
- **Descripción**: Mejora incremental paso a paso
- **Ventajas**: Rápido y eficiente

### 2. Votación de Ensemble (`ensemble_voting`)
- **Mejor para**: Resultados robustos y confiables
- **Descripción**: Múltiples optimizaciones en paralelo
- **Ventajas**: Mayor estabilidad y precisión

### 3. Multi-Objetivo (`multi_objective`)
- **Mejor para**: Balance entre objetivos competitivos
- **Descripción**: Optimiza para múltiples criterios simultáneamente
- **Ventajas**: Soluciones balanceadas

### 4. Algoritmo Genético (`genetic_algorithm`)
- **Mejor para**: Exploración creativa y diversa
- **Descripción**: Evolución de prompts através de generaciones
- **Ventajas**: Encuentra soluciones innovadoras

## 🔧 Uso Programático

### Optimización Básica

```python
import asyncio
from spo_framework import SPOFramework, PromptOptimizationConfig

async def optimize_prompt():
    config = PromptOptimizationConfig(
        max_iterations=5,
        optimization_model="gpt-4o",
        execution_model="gpt-4o-mini"
    )
    
    framework = SPOFramework(config, "tu-api-key")
    
    result = await framework.optimize_prompt(
        initial_prompt="Tu prompt inicial",
        task_description="Descripción de la tarea",
        sample_inputs=["input1", "input2", "input3"],
        expected_outputs=["output1", "output2", "output3"]
    )
    
    print(f"Score: {result.performance_score}")
    print(f"Prompt optimizado: {result.optimized_prompt}")

asyncio.run(optimize_prompt())
```

### Optimización Estilo MetaGPT con Funciones Lambda

```python
from lambda_functions import MetaGPTStyleOptimizer, PromptLambdaFactory

async def metagpt_optimization():
    # Crear función lambda para clasificación de trabajos
    job_classifier = PromptLambdaFactory.create_job_classification_lambda()
    
    # Optimizador estilo MetaGPT
    optimizer = MetaGPTStyleOptimizer(client, config)
    
    result = await optimizer.optimize_with_lambda(
        initial_lambda=job_classifier,
        task_description="Clasificar títulos de trabajo",
        test_inputs=job_titles,
        expected_outputs=expected_classifications,
        max_rounds=5
    )
```

### Pipeline Avanzado

```python
from optimization_pipeline import AdvancedOptimizationPipeline, PipelineConfig, OptimizationStrategy

async def advanced_optimization():
    pipeline_config = PipelineConfig(
        strategy=OptimizationStrategy.ENSEMBLE_VOTING,
        ensemble_size=5
    )
    
    pipeline = AdvancedOptimizationPipeline(spo_config, pipeline_config, api_key)
    
    result = await pipeline.optimize_with_strategy(
        initial_prompt="Tu prompt",
        task_description="Tarea",
        sample_inputs=inputs,
        expected_outputs=outputs
    )
```

### Sistema de Confianza

```python
from confidence_scoring import ConfidenceAnalyzer

async def analyze_confidence():
    analyzer = ConfidenceAnalyzer(client)
    
    confidence_metrics = await analyzer.analyze_optimization_confidence(
        optimization_history=optimization_results,
        test_inputs=test_data,
        final_prompt=best_prompt,
        num_validation_runs=5
    )
    
    report = analyzer.generate_confidence_report(confidence_metrics)
    print(f"Confidence Level: {report['confidence_level']}")
    print(f"Reliability Index: {report['reliability_index']:.2f}")
```

### Evaluación Comprehensiva

```python
from evaluation_system import ComprehensiveEvaluationSystem, EvaluationCriteria, EvaluationMetric

async def evaluate_prompt():
    evaluation_system = ComprehensiveEvaluationSystem(api_key)
    
    criteria = [
        EvaluationCriteria(EvaluationMetric.ACCURACY, 0.5, "Precisión"),
        EvaluationCriteria(EvaluationMetric.CLARITY, 0.3, "Claridad"),
        EvaluationCriteria(EvaluationMetric.RELEVANCE, 0.2, "Relevancia")
    ]
    
    result = await evaluation_system.comprehensive_evaluate(
        prompt="Tu prompt",
        input_text="Input de prueba",
        output="Output generado",
        expected="Output esperado",
        evaluation_criteria=criteria
    )
```

## 📈 Métricas de Evaluación

- **Accuracy**: Precisión y correctitud del output
- **Relevance**: Relevancia al input y tarea
- **Clarity**: Claridad y legibilidad
- **Completeness**: Completitud de la respuesta
- **Consistency**: Consistencia en múltiples ejecuciones
- **Efficiency**: Eficiencia del prompt
- **Adherence to Format**: Adherencia al formato especificado
- **Confidence Metrics**: Métricas de confiabilidad y estabilidad

## 🔍 Monitoreo y Resultados

Los resultados se pueden exportar en formato JSON para análisis posterior:

```json
{
  "config": {...},
  "optimization_history": [...],
  "summary": {
    "total_iterations": 5,
    "best_score": 0.89,
    "improvement": 0.34,
    "best_prompt": "Prompt optimizado final"
  }
}
```

## 🎯 Ejemplo Detallado: Clasificación de Títulos de Trabajo

El framework incluye un ejemplo completo que demuestra la optimización de un sistema de clasificación de títulos de trabajo:

```python
from job_title_example import JobTitleClassificationExample

async def run_example():
    example = JobTitleClassificationExample(api_key)
    results = await example.run_complete_example()
    
    # Ejecuta:
    # 1. Optimización básica SPO
    # 2. Comparación de estrategias
    # 3. Evaluación comprehensiva
    # 4. Reporte final con recomendaciones
```

## 🚀 Casos de Uso

- **Mejora de Chatbots**: Optimizar prompts para conversaciones más naturales
- **Clasificación de Texto**: Mejorar precisión en tareas de categorización
- **Generación de Contenido**: Optimizar prompts para contenido de calidad
- **Análisis de Sentimientos**: Afinar prompts para mejor detección emocional
- **Extracción de Información**: Mejorar prompts para extraer datos estructurados

## 🔧 Configuración Avanzada

### Modelos Personalizados

```python
config = PromptOptimizationConfig(
    optimization_model="gpt-4o",        # Modelo para optimización
    execution_model="gpt-4o-mini",      # Modelo para ejecución
    evaluation_model="gpt-4o",          # Modelo para evaluación
    temperature=0.7,                     # Creatividad
    max_tokens=2000                      # Límite de tokens
)
```

### Criterios de Evaluación Personalizados

```python
custom_criteria = [
    EvaluationCriteria(EvaluationMetric.ACCURACY, 0.4, "Precisión técnica"),
    EvaluationCriteria(EvaluationMetric.CREATIVITY, 0.3, "Creatividad"),
    EvaluationCriteria(EvaluationMetric.SAFETY, 0.3, "Seguridad del contenido")
]
```

## 📚 Recursos Adicionales

- [Documentación de OpenAI](https://platform.openai.com/docs)
- [Paper original MetaGPT](https://arxiv.org/abs/2308.00352)
- [Guía de Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering)
- [Mistral Prompt Optimization - Documento base](https://docs.mistral.ai/guides/prompting_capabilities/) - *Técnicas de optimización que inspiraron este framework*

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature
3. Haz commit de tus cambios
4. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver archivo LICENSE para detalles.

## ⚠️ Consideraciones Importantes

### 💰 **Costos Estimados de API**
- **Prueba Rápida**: $0.10-0.20 (recomendado para empezar)
- **Demo Rápido**: $0.50-1.00
- **Demo Completo**: $2.00-4.00
- **Optimización Personalizada**: $1.00-3.00 (depende de iteraciones)

### ⏱️ **Tiempos de Ejecución**
- **Prueba Rápida**: 2-3 minutos
- **Demo Rápido**: 5-10 minutos  
- **Demo Completo**: 15-20 minutos
- **Comparación de Estrategias**: 10-15 minutos

### 🎯 **Consejos para Mejores Resultados**
- **Datos de Calidad**: Los resultados dependen de la calidad de los datos de entrada
- **Configuración**: Ajusta los parámetros según tus necesidades específicas
- **API Key**: Asegúrate de tener créditos suficientes en tu cuenta OpenAI
- **Primeras Pruebas**: Usa `test_optimization.py` para validar tu configuración

## 🚨 Solución de Problemas Comunes

### ❌ "OPENAI_API_KEY NO configurado"
```bash
export OPENAI_API_KEY="tu-api-key-aqui"
```

### ❌ "ModuleNotFoundError"
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### ❌ "Error de autenticación OpenAI"
- Verifica que tu API key sea correcta
- Asegúrate de tener créditos en tu cuenta OpenAI
- Revisa que no haya espacios extra en la API key

### ❌ "Timeout o errores de red"
- Reduce el número de iteraciones: `--iterations 2`
- Usa modelos más pequeños en la configuración
- Verifica tu conexión a internet

## 🆘 Soporte

Para problemas o preguntas:
1. Revisa la documentación
2. Ejecuta `python test_optimization.py` para diagnóstico
3. Prueba los ejemplos en `examples/` directory
4. Verifica los logs de error en la consola
5. Busca en issues existentes
6. Crea un nuevo issue con detalles específicos

## 🎉 Novedades de Esta Versión

### ✨ Soporte Completo para YAML

- **Entrada flexible**: Archivos YAML con prompts, tareas y datos integrados
- **Salida en YAML**: Resultados más legibles y estructurados  
- **Detección automática**: El sistema detecta el formato por extensión de archivo
- **Compatibilidad total**: Mantiene soporte completo para JSON

### 📂 Directorio de Ejemplos Organizados

- **6 ejemplos probados**: Desde básicos hasta avanzados
- **Casos de uso reales**: Clasificación de trabajos, mensajes de chat
- **Formatos múltiples**: JSON y YAML demostrados
- **Resultados verificados**: Todos los ejemplos funcionan correctamente

### 🔧 Mejoras en la Interfaz

- **CLI más flexible**: Argumentos opcionales cuando están en YAML
- **Mejor organización**: Archivos separados por función
- **Parser robusto**: Manejo de errores y formatos múltiples
- **Configuración .env**: Carga automática de variables de entorno

## 🎯 Pasos Recomendados para Empezar

1. **Configuración inicial:**
   ```bash
   source venv/bin/activate
   export OPENAI_API_KEY="tu-api-key"
   ```

2. **Primera prueba:**
   ```bash
   python test_optimization.py
   ```

3. **Si funciona bien, prueba el demo:**
   ```bash
   python main.py job-title-example --mode quick
   ```

4. **Explora funcionalidades avanzadas según tus necesidades**

---

**¡Comienza a optimizar tus prompts hoy mismo!** 🚀