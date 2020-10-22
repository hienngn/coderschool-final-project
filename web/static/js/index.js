const btn = document.getElementById('submitBtnEn');
console.log("yay")

btn.addEventListener('click', function (e){
  const fileInput = document.getElementById('file');
  const formData  = new FormData();
  formData.append('file', fileInput.files[0]);
  formData.append()

  fetch('/predict/', {
    method: 'POST',
    body: formData,
  }).then(res => res.json()).then(data => {
    console.log(data);
    if (!data) {
      return
    }
    //document.getElementById('result').innerHTML=data['hip'];
    document.getElementById('prodpic').src = data['filename'];
    document.getElementById('prodpic').style.display = "inline-block";
  })
    .catch(error => {
      document.getElementById('result').innerHTML="";
      document.getElementById('prodpic').src = "";
      document.getElementById('prodpic').style.display = "none";
      console.error('Error:', error);
    });
})

