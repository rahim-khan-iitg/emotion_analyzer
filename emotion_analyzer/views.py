from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from src.pipelines.prediction_pipelines import PredictPipeline
@csrf_exempt
def home(request):
    if request.method=="POST":
        prediction=PredictPipeline()
        text=request.POST.get("text")
        pred=prediction.Predict(text)
        result="Positive"
        if pred<0.5:
            result="Negative"
        return render(request,"index.html",{"result":result})
    return render(request,"index.html")