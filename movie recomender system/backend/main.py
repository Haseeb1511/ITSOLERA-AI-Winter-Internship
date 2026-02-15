from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from backend.recommender import recommend, get_all_movies

app = FastAPI(title="Movie Recommender API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Movie Recommender API is running"}


@app.get("/movies")
def movies():
    """Return all movie titles for autocomplete."""
    return {"movies": get_all_movies()}


@app.get("/recommend")
def get_recommendations(
    movie: str = Query(..., description="Movie title to get recommendations for"),
    top_k: int = Query(5, description="Number of recommendations", ge=1, le=20),
):
    """Get movie recommendations based on content similarity."""
    results = recommend(movie, top_k)
    if results is None:
        return {"error": "Movie not found", "movies": []}
    return {"movie": movie, "recommendations": results}
