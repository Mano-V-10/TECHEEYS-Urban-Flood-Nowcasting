# TECHEEYS Urban Flood Nowcasting System — Project Sources

This repository combines the materials supplied for the TECHEEYS project.

## Included source materials

1. **AI Studio exported application** — stored in `app/`.
   - React/Vite frontend source
   - FastAPI backend source
   - Project configuration and dependency files
   - The original exported ZIP is preserved in `source-materials/AI-Studio-Original-Export.zip`.

2. **Standalone HTML demonstration** — stored in `demo/advanced_weather_prediction.html`.
   - This is the earlier standalone weather/flood prediction prototype supplied for the project.
   - It is retained as a supplementary demo/reference and is not silently presented as the same implementation as the AI Studio application.

3. **Demo video** — `demo/Urban-Flood-Nowcasting-Demo.mp4`.

4. **Presentation** — `presentation/Urban-Flood-Nowcasting-System.pptx`.

## Important implementation note

The AI Studio application contains its own flood-nowcasting implementation and a FastAPI/Scikit-learn backend. The standalone HTML file contains a separate browser-side prototype. Both are retained because they were supplied as project materials.

The presentation describes a proposed Random Forest approach and reports approximately 89% accuracy on simulated test data. This repository does not claim that figure as real-world operational accuracy.
