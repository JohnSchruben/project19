[turn-angle-tests]
run 'dotnet test' in turn-angle-tests directory

[ollama-turn-test]
download and install ollama https://ollama.com/download/windows

then run:
ollama serve
ollama pull llava

run dotnet run '.\Ollama Instruction Angle Test.cs' in ollama-turn-test directory
