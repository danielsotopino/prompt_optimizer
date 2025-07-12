#!/usr/bin/env python3
"""
Script de prueba rápida para el framework SPO
Ejecuta una optimización básica sin necesidad de configuración compleja
"""
import asyncio
import os
import sys
from spo_framework import SPOFramework, PromptOptimizationConfig

async def quick_test():
    """Prueba rápida del framework"""
    
    # Verificar API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: Por favor configura tu OPENAI_API_KEY")
        print("Ejecuta: export OPENAI_API_KEY='tu-api-key-aqui'")
        return
    
    print("🚀 Iniciando prueba de optimización SPO...")
    print("=" * 50)
    
    # Configuración de prueba
    config = PromptOptimizationConfig(
        max_iterations=3,  # Solo 3 iteraciones para prueba rápida
        optimization_model="gpt-4o-mini",  # Modelo más económico
        execution_model="gpt-4o-mini",
        evaluation_model="gpt-4o-mini",
        temperature=0.7
    )
    
    # Datos de prueba
    initial_prompt = """Clasifica el siguiente título de trabajo en categorías apropiadas.
Usa el formato: Categoría - Subcategoría

Título de trabajo: {job_title}
Clasificación:"""
    
    task_description = "Clasificar títulos de trabajo en categorías significativas"
    
    sample_inputs = [
        "Senior Software Engineer",
        "Data Scientist", 
        "Marketing Manager"
    ]
    
    expected_outputs = [
        "Technology - Software Engineering",
        "Technology - Data Science",
        "Business - Marketing"
    ]
    
    print("📝 Prompt inicial:")
    print(f"'{initial_prompt}'")
    print("\n📊 Datos de prueba:")
    for i, (inp, exp) in enumerate(zip(sample_inputs, expected_outputs)):
        print(f"  {i+1}. '{inp}' → '{exp}'")
    
    print("\n🔄 Ejecutando optimización...")
    
    try:
        # Crear framework y ejecutar optimización
        framework = SPOFramework(config, api_key)
        
        result = await framework.optimize_prompt(
            initial_prompt=initial_prompt,
            task_description=task_description,
            sample_inputs=sample_inputs,
            expected_outputs=expected_outputs
        )
        
        # Mostrar resultados
        print("\n✅ ¡Optimización completada!")
        print("=" * 50)
        print(f"📈 Puntuación final: {result.performance_score:.2f}")
        print(f"🔄 Iteraciones realizadas: {result.iteration}")
        print(f"⏱️ Tiempo de ejecución: {result.execution_time:.1f}s")
        
        print(f"\n📝 Prompt optimizado:")
        print("-" * 30)
        print(result.optimized_prompt)
        print("-" * 30)
        
        print(f"\n💭 Retroalimentación:")
        print(result.feedback)
        
        # Resumen de la optimización
        summary = framework.get_optimization_summary()
        print(f"\n📊 Resumen de optimización:")
        print(f"  • Mejora total: +{summary['improvement']:.2f}")
        print(f"  • Tiempo total: {summary['total_time']:.1f}s")
        print(f"  • Mejor iteración: #{summary['best_iteration']}")
        
        print("\n🎉 ¡Prueba completada exitosamente!")
        
    except Exception as e:
        print(f"\n❌ Error durante la optimización: {str(e)}")
        print("💡 Asegúrate de que tu API key sea válida y tengas créditos disponibles")

if __name__ == "__main__":
    print("🤖 Framework SPO - Prueba Rápida")
    print("Asegúrate de haber configurado OPENAI_API_KEY")
    print()
    
    # Ejecutar prueba
    asyncio.run(quick_test())