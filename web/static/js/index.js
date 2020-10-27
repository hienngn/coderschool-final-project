const enBtn = document.getElementById('submitBtnEn');
const vnBtn = document.getElementById('submitBtnVn');

enBtn.addEventListener('click', function (e){
  callPredict("English");
})
vnBtn.addEventListener('click', function (e){
  callPredict("Vietnamese");
})

function callPredict(lang) {
  const fileInput = document.getElementById('file');
  const dewarpOpt = document.querySelector('#dewarp');
  const isDewarp = (dewarpOpt.checked) ? dewarpOpt.checked : false;
  const formData  = new FormData();
  formData.append('file', fileInput.files[0]);
  if (isDewarp) {
    formData.append('dewarp', true);
  } else {
    formData.append('dewarp', false);
  }
  // Language
  formData.append('lang', lang);

  showInfoBar();
  $('#result').empty();
  $('#scorebox').hide();
  $('#scorearea').empty();
  $('#suggestionsbox').empty();
  document.getElementById('prodpic').src = "";
  fetch('/predict/', {
    method: 'POST',
    body: formData,
  }).then(res => res.json()).then(data => {
    console.log(data);
    if (!data) {
      hideInfoBar();
      return
    }
    // document.getElementById('result').innerHTML=genIngTable(res);
    $('#result').append(genIngTable(data['res']));
    $('#scorebox').show();
    $('#scorearea').append(data['score']);
    $('#suggestionsbox').append(genProds(data['recom']));
    document.getElementById('prodpic').src = data['filename'];
    document.getElementById('prodpic').style.display = "inline-block";
    hideInfoBar();
  })
    .catch(error => {
      $('#result').empty();
      $('#scorebox').hide();
      $('#scorearea').empty();
      $('#suggestionsbox').empty();
      document.getElementById('prodpic').src = "";
      document.getElementById('prodpic').style.display = "none";
      hideInfoBar();
      console.error('Error:', error);
    });
}

// GENERATE TABLE
function genIngTable(res) {
  if (!res || res.length < 1) {
    return "";
  }
  var table = $("<table></table>").addClass(['collapse', 'ba', 'br2', 'b--black-10', 'pv2', 'ph3', 'bg-white']);
  var thead = $(`<thead><tr>
      <th class='pv2 ph3 tl f6 fw6 ttu'>Name</th>
      <th class='pv2 ph3 tl f6 fw6 ttu'>Irritancy</th>
      <th class='pv2 ph3 tl f6 fw6 ttu'>Comedogenicity</th>
      <th class='pv2 ph3 tl f6 fw6 ttu'>Rating</th>
      <th class='pv2 ph3 tl f6 fw6 ttu'>Functions</th>
    </tr></thead>`);
      //<th class='pv2 ph3 tl f6 fw6 ttu'>Description</th>
      // <th class='pv2 ph3 tl f6 fw6 ttu'>Quick Facts</th>
  table.append(thead)

  var tbody = $('<tbody></tbody>')
  var curr, cnode, cth;
  for (var i=0;i<res.length;i++) {
    curr = res[i];
    cnode = $('<tr></tr>').addClass(['striped--light-gray', 'f6']);
    cnode.append($('<td></td>').addClass(['pv2', 'ph3']).text(curr['Ingredient_name']));
    //cnode.append($('<td></td>').addClass(['pv2', 'ph3']).text(curr['Quick_facts']));
    cnode.append($('<td></td>').addClass(['pv2', 'ph3']).text(curr['Irritancy']));
    cnode.append($('<td></td>').addClass(['pv2', 'ph3']).text(curr['Comedogenicity']));
    cnode.append($('<td></td>').addClass(['pv2', 'ph3']).text(curr['Rating']));
    cnode.append($('<td></td>').addClass(['pv2', 'ph3']).text(curr['Functions']));
    //cnode.append($('<td></td>').addClass(['pv2', 'ph3']).text(curr['Description']));
    tbody.append(cnode);
  }
  table.append(tbody);
  return table;
}

// SUGGESTIONS CODE
function genProds(products) {
  if (!products || products.length < 1) {
    return "";
  }
  var anode = $('<div><div class="w-100 tc"><h1 class="ttu tracked f1">Suggestions for alternative products</h1></div></div>').addClass(['w-100', 'flex', 'flex-wrap']);
  for (var i=0;i < products.length; i++) {
    anode.append(productCard(products[i]));
  }
  return anode
}

function productCard(prod) {
  var prodHTML = `
<article class="br2 ba dark-gray b--black-10 mv4 w-100 w-50-m w-25-l mw5 center">
  <a target='_blank' href='${prod['product_url']}'><img src="${prod['product_img']}" class="db w-100 br2 br--top" alt="${prod['product_brand'] + " " + prod['product_name']}">
  <div class="pa2 ph3-ns pb3-ns">
    <div class="dt w-100 mt1">
      <div class="dtc">
        <a target='_blank' class='blu dim link' href='${prod['product_url']}'><h1 class="f5 f4-ns mv0">${prod['product_brand'] + " " + prod['product_name']}</h1></a>
      </div>
      <div class="dtc tr">
        <h2 class="f5 mv0">Score: ${prod['product_score']}</h2>
      </div>
    </div>
    <p class="f6 lh-copy measure mt2 mid-gray"><span class='ttu'>Ingredients:</span> ${prod['ingredient_list']}</p>
  </div>
</article>
  `
  return prodHTML;
}

// INFO BAR CODE - bottom of the screen
const infobar = document.getElementById('infobar');
function showInfoBar() {
  infoBarTxtElem.innerHTML = "Uploading image ...";
  inst = setInterval(utb(), 5000);
  infobar.style.display = "block";
}

function hideInfoBar() {
  if (inst) {
    clearInterval(inst);
    inst = null;
  }
  infobar.style.display = "none";
}

var messages  = [
  "Calculating text points ...",
  "Converting points to spans ...",
  "Localizing text areas ...",
  "Preprocessing image ...",
  "Calculating opencv threshold ...",
  "Cropping text areas ...",
  "Running OCR ...",
  "Calculating total score ...",
  "Finding recommendations ...",
  "Results coming right up ...",
  "Holy crap this image is difficult ...",
  "This is taking longer than usual ...",
];
var infoBarTxtElem = document.getElementById("infobar_msg")
var inst = null;

function utb() {
  var counter = 0;
  return function() {
    infoBarTxtElem.innerHTML = messages[counter];
    counter++;
    if (counter >= messages.length) {
      counter = 0;
      if (inst) {
        clearInterval(inst);
      }
    }
  }
}
