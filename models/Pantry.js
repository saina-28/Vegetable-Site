const mongoose = require('./connection')

const pantrySchema = new mongoose.Schema({
  img : String,
  item_name : String,
  item_category : String,
  item_available_qt : Number,
  item_price : Number,
})

const Pantry = mongoose.model('Pantry', pantrySchema)

module.exports = Pantry;
