from sklearn.feature_extraction.text import TfidfVectorizer
 
 
_NEWS_STOPWORDS = [
    "said", "says", "say", "told", "according", "reported",
    "added", "noted", "stated", "announced",
    "new", "york", "times", "mr", "ms", "mrs", "dr",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "january", "february", "march",
    "april", "june", "july", "august", "september",
    "october", "november", "december",
    # Generic filler
    "also", "one", "two", "three", "year", "years",
    "people", "time", "day", "week", "month",
    "percent", "including", "officials", "government",
]
 
 
def create_embeddings(docs: list[dict]):
    """
    Build a TF-IDF matrix from a list of chunk dicts.
 
    Vectorizer design choices for a news corpus
    --------------------------------------------
    sublinear_tf=True
        Apply log(1 + tf) instead of raw tf.
        Stops a word that appears 50× from getting 50× the weight
        of one that appears once — crucial for long news articles.
 
    max_df=0.85
        Ignore terms that appear in more than 85% of chunks.
        Catches corpus-wide filler ("said", "new") that survive the
        stop-word list but have no discriminative power.
 
    min_df=3
        Ignore terms that appear in fewer than 3 chunks.
        Filters OCR artifacts, typos, and overly rare tokens that
        would otherwise create noisy singleton IDF columns.
 
    ngram_range=(1, 2)
        Index both single words and bigrams.
        "climate change", "interest rate", "ukraine war" are now
        single features, making query matching far more precise.
 
    max_features=60_000
        Cap vocabulary size to keep the matrix manageable in memory.
        60k covers a large news corpus comfortably; lower to 30k if
        RAM is tight.
 
    stop_words
        sklearn's built-in English list + custom news boilerplate.
    """
    texts = [doc["text"] for doc in docs]
 
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        max_df=0.85,
        min_df=2,
        ngram_range=(1, 2),
        max_features=60_000,
        stop_words=_get_stopwords(),
    )
 
    embeddings = vectorizer.fit_transform(texts)
 
    print(f"[embed] Vocabulary size : {len(vectorizer.vocabulary_):,}")
    print(f"[embed] Matrix shape    : {embeddings.shape}")
 
    return embeddings, vectorizer
 
 
def _get_stopwords() -> list[str]:
    """Merge sklearn's English stop-word list with our news-specific additions."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return list(set(ENGLISH_STOP_WORDS) | set(_NEWS_STOPWORDS))
