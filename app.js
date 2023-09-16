require('dotenv').config()
const express = require('express')

const path = require('path')
const cors = require('cors')
const session = require('express-session');
const cookieParser = require('cookie-parser')
const app = express()
const port = 3000
app.set("view engine", "ejs")


const connection = require('./models/connection')
const authRoute = require('./routes/auth');
const secure = require('./middlewares/auth');
const alreadylogged = require('./middlewares/alreadylogged');
const pantryRoute = require('./routes/pantry')
const userRoute = require("./routes/user")
const cartRoute = require("./routes/cart")
const checkoutRoute = require("./routes/checkout")


app.use(express.static(path.join(__dirname, 'public')));
app.use((req, res, next) => {
  console.log("HTTP Method - " + req.method + ", URL - "+ req.url);
  next();
})


app.use(express.urlencoded({ extended: true }));
app.use(express.json({
  limit : '50mb',
  verify: function (req, res, buf) {
    if (req.originalUrl.startsWith('/checkout/web')) {
      req.rawBody = buf.toString();
    }
  },
}))
app.use(cookieParser())
app.use(session({
  secret : "secret",
  resave : false,
  saveUninitialized : false,
  
  cookie: { maxAge: 60*60*1000 }
}))
app.use(cors())

app.use("/checkout", checkoutRoute)
app.use("/auth", authRoute)
app.use("/pantry", pantryRoute)
app.use("/user", userRoute)
app.use("/cart", cartRoute)

const isAuthenticated = (req)=>{
  if(req.cookies.userid){
    return 1;
  }
  return 0;
}

app.get('/', alreadylogged, (req, res) => {
  res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  res.render("index")
})

app.get('/dashboard', secure, (req, res) => {
  res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  res.render("pages/dashboard")
})

app.get('/user', secure, (req, res) => {
  res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  res.render("pages/user")
})

app.get('/cart', secure, (req, res) => {
  res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  res.render("pages/cart")
})

app.get('/about', (req, res) => {
  let auth = isAuthenticated(req)
  res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  res.render("pages/about",{ auth })
})  

app.get('/contact', (req, res) => {
  let auth = isAuthenticated(req)
  res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  res.render("pages/contact", { auth })
})


app.get('*', (req, res) =>{
  res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  res.redirect("/");
});

app.post('*', (req, res) =>{
  res.status(400).json({ message : "This route does not exist!" });
});


app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
})