async function classify(filename) {
  const res = await fetch("https://genre-identifier-5ahb.onrender.com/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename })
  });

  const data = await res.json();

  const high_predictions = {
  "blues.00054.wav": "blues",
  "blues.00066.wav": "blues",
  "blues.00076.wav": "blues",
  "classical.00020.wav": "classical",
  "classical.00068.wav": "classical",
  "classical.00098.wav": "classical",
  "country.00009.wav": "country",
  "country.00054.wav": "country",
  "country.00075.wav": "country",
  "disco.00012.wav": "disco",
  "disco.00028.wav": "disco",
  "disco.00063.wav": "disco",
  "hiphop.00008.wav": "hiphop",
  "hiphop.00017.wav": "hiphop",
  "hiphop.00079.wav": "hiphop",
  "jazz.00014.wav": "jazz",
  "jazz.00026.wav": "jazz",
  "jazz.00070.wav": "jazz",
  "metal.00035.wav": "metal",
  "metal.00039.wav": "metal",
  "metal.00087.wav": "metal",
  "pop.00021.wav": "pop",
  "pop.00043.wav": "pop",
  "pop.00061.wav": "pop",
  "reggae.00001.wav": "disco",
  "reggae.00026.wav": "reggae",
  "reggae.00065.wav": "reggae",
  "rock.00017.wav": "rock",
  "rock.00041.wav": "rock",
  "rock.00074.wav": "rock",
  };

  document.getElementById("prediction").innerText = data.prediction;
  document.getElementById("actual").innerText = data.actual;
  document.getElementById("baseline").innerText = data.low_baseline;
  // document.getElementById("high").innerText = data.high_baseline;
  document.getElementById("high").innerText = high_predictions[filename];
  document.getElementById("classified-file").innerText = filename;

  const audio = document.getElementById("audio-player");
  audio.src = `./GTZAN/genres_original/${filename.split('.')[0]}/${filename}`;
  audio.load();
}