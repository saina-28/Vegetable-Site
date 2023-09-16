const mongoose = require('./connection')

const UserSchema = new mongoose.Schema({
  name : {
    type : String,
    default: "User",
  },
  email :{
    type : String,
    required : true,
    unique : true,
  },
  password :{
    type : String,
    required : true,
  },
  img : {
    type : String,
    default: "../assets/default-user.png",
  },
  address : {
    type : String,
    default: "",
  },
  mob : {
    type : String,
    default: "",
  },
  gender : {
    type : String,
    default: "",
  },
  dob : {
    type : Date,
  },
  stripe_customer : {
    type : String,
    default : null
  }
})

const User = mongoose.model('User', UserSchema)

module.exports = User;
