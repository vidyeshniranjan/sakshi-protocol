decision = decide(state, distortion)

if decision == "retrieve":
    context = retrieve(prompt)
    output = generator.generate(prompt + context)
