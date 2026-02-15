import { useState, useEffect, useRef } from 'react'
import './App.css'

const API = '/api'

function App() {
    const [query, setQuery] = useState('')
    const [allMovies, setAllMovies] = useState([])
    const [suggestions, setSuggestions] = useState([])
    const [recommendations, setRecommendations] = useState([])
    const [selectedMovie, setSelectedMovie] = useState('')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')
    const [showSuggestions, setShowSuggestions] = useState(false)
    const [activeIndex, setActiveIndex] = useState(-1)
    const wrapperRef = useRef(null)

    // Fetch all movie titles on mount
    useEffect(() => {
        fetch(`${API}/movies`)
            .then((res) => res.json())
            .then((data) => setAllMovies(data.movies || []))
            .catch(() => setAllMovies([]))
    }, [])

    // Close dropdown on outside click
    useEffect(() => {
        function handleClick(e) {
            if (wrapperRef.current && !wrapperRef.current.contains(e.target)) {
                setShowSuggestions(false)
            }
        }
        document.addEventListener('mousedown', handleClick)
        return () => document.removeEventListener('mousedown', handleClick)
    }, [])

    // Filter suggestions as user types
    function handleChange(e) {
        const value = e.target.value
        setQuery(value)
        setActiveIndex(-1)

        if (value.trim().length < 1) {
            setSuggestions([])
            setShowSuggestions(false)
            return
        }

        const filtered = allMovies
            .filter((m) => m.toLowerCase().includes(value.toLowerCase()))
            .slice(0, 8)

        setSuggestions(filtered)
        setShowSuggestions(filtered.length > 0)
    }

    // Keyboard navigation
    function handleKeyDown(e) {
        if (!showSuggestions) {
            if (e.key === 'Enter' && query.trim()) {
                fetchRecommendations(query.trim())
            }
            return
        }

        if (e.key === 'ArrowDown') {
            e.preventDefault()
            setActiveIndex((prev) => (prev < suggestions.length - 1 ? prev + 1 : 0))
        } else if (e.key === 'ArrowUp') {
            e.preventDefault()
            setActiveIndex((prev) => (prev > 0 ? prev - 1 : suggestions.length - 1))
        } else if (e.key === 'Enter') {
            e.preventDefault()
            if (activeIndex >= 0 && activeIndex < suggestions.length) {
                selectMovie(suggestions[activeIndex])
            } else if (query.trim()) {
                fetchRecommendations(query.trim())
                setShowSuggestions(false)
            }
        } else if (e.key === 'Escape') {
            setShowSuggestions(false)
        }
    }

    function selectMovie(title) {
        setQuery(title)
        setShowSuggestions(false)
        setActiveIndex(-1)
        fetchRecommendations(title)
    }

    async function fetchRecommendations(movie) {
        setLoading(true)
        setError('')
        setRecommendations([])
        setSelectedMovie(movie)

        try {
            const res = await fetch(
                `${API}/recommend?movie=${encodeURIComponent(movie)}&top_k=5`
            )
            const data = await res.json()

            if (data.error) {
                setError(`"${movie}" not found. Try a different title.`)
            } else {
                setRecommendations(data.recommendations || [])
            }
        } catch {
            setError('Could not reach the server. Is the backend running?')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="app">
            <header className="header">
                <h1>
                    🎬 Movie <span className="accent">Recommender</span>
                </h1>
                <p>Enter a movie you like and get similar recommendations</p>
            </header>

            <div className="search-wrapper" ref={wrapperRef}>
                <input
                    className="search-input"
                    type="text"
                    placeholder="Search for a movie…"
                    value={query}
                    onChange={handleChange}
                    onKeyDown={handleKeyDown}
                    onFocus={() => suggestions.length > 0 && setShowSuggestions(true)}
                />

                {showSuggestions && (
                    <ul className="suggestions">
                        {suggestions.map((title, i) => (
                            <li
                                key={title}
                                className={i === activeIndex ? 'active' : ''}
                                onMouseDown={() => selectMovie(title)}
                            >
                                {title}
                            </li>
                        ))}
                    </ul>
                )}
            </div>

            {loading && (
                <div className="loader">
                    <div className="spinner" />
                </div>
            )}

            {error && <p className="message error">{error}</p>}

            {recommendations.length > 0 && (
                <div>
                    <p className="results-title">
                        Similar to "{selectedMovie}"
                    </p>
                    <div className="results-grid">
                        {recommendations.map((movie, i) => (
                            <div
                                className="movie-card"
                                key={movie.id}
                                onClick={() => selectMovie(movie.title)}
                                style={{ cursor: 'pointer' }}
                            >
                                <span className="card-index">{i + 1}</span>
                                <span className="card-title">{movie.title}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}

export default App
