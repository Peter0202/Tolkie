import requests

url = "http://localhost:5000/generate"
payload = {
    "model": "fietje",
    "prompt_version": "0",
    "text": "Over longgeneeskunde Of het nu gaat om astma, COPD, bronchiëctasieën of een uitgebreid revalidatieprogramma: op de polikliniek longgeneeskunde van Máxima MC (MMC) bieden we de beste zorg voor u. De poli bevindt zich zowel in Eindhoven als Veldhoven. Onze longartsen zorgen voor snelle toegankelijke en persoonlijke begeleiding. Alle specialisten kennen hun eigen aandachtsgebied, waardoor zij u met specifieke aandoeningen goed kunnen helpen. Er zijn korte lijnen met huisartsen, fysiotherapeuten, diëtisten, longverpleegkundigen en specialisten, waardoor we een multidisciplinaire behandeling aanbieden. Daarnaast werken we ook veel samen met universitaire centra. Op deze website vindt u meer informatie over aandoeningen, symptomen, oorzaken, behandelingen en hoe u een afspraak maakt op de polikliniek longgeneeskunde"
}
headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
