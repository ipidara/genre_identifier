async function classify(filename) {
  const res = await fetch("https://genre-identifier-5ahb.onrender.com/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename })
  });

  const data = await res.json();

  document.getElementById("prediction").innerText = data.prediction;
  document.getElementById("actual").innerText = data.actual;
  document.getElementById("baseline").innerText = data.low_baseline;
  document.getElementById("classified-file").innerText = filename;

  const audio = document.getElementById("audio-player");
  audio.src = `./GTZAN/genres_original/${filename.split('.')[0]}/${filename}`;
  audio.load();
}